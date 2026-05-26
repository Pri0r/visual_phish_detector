import os
import json
import difflib
import torch
import numpy as np
import faiss
import easyocr
import warnings
from PIL import Image, ImageStat
from transformers import CLIPProcessor, CLIPVisionModel
from ultralytics import YOLO
from rdflib import Graph, Namespace
from rdflib.namespace import RDF

from utils import check_domain_consistency

# fitlayout ontology
B = Namespace("http://fitlayout.github.io/ontology/render.owl#")

# ignore DecompressionBombError, future warnings
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=FutureWarning)


def get_text_similarity(text1, text2):
    """Calculates similarity between two strings"""
    if not text1 or not text2: return 0.0
    return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def make_square_with_padding(img_rgb, background_color):
    """Converts image into a square, with the smaller rectangle side being filled by a background color"""
    width, height = img_rgb.size
    # already a square
    if width == height: 
        return img_rgb
    
    size = max(width, height)
    new_img = Image.new("RGB", (size, size), background_color)
    new_x = (size - width) // 2
    new_y = (size - height) // 2
    new_img.paste(img_rgb, (new_x, new_y))
    return new_img


def calculate_box_score(x, y, w, h, img_height, img_width, tag = None):
    """Calculates a logo bounding box score, more favorable locations/sizes have higher score"""
    if w == 0 or h == 0: return None
    
    area = w * h
    # box shouldnt be too small or too big
    if w < 20 or h < 15: return None
    if area > 80000: return None 

    # width/height ratio shouldnt be too extreme
    ar = w / h
    if ar < 0.2 or ar > 10: return None
        
    # the higher/more left the better
    pos_y_score = max(0.0, 1.0 - (y / (img_height / 1.33)))
    pos_x_score = max(0.0, 1.0 - 0.4 * (x / img_width)) 
    size_score = min(area / 15000.0, 1.0)
    
    # favor image elements
    if tag in ["IMG", "PICTURE"]: tag_bonus = 3.0
    elif tag in ["SVG"]: tag_bonus = 2.0
    elif tag == "A": tag_bonus = 1.2
    else: tag_bonus = 0.5
        
    # final score
    return pos_y_score * pos_x_score * size_score * tag_bonus


class LogoClassifier:
    """Logo phishing classifier"""
    def __init__(
        self,
        db_vectors_file,
        db_metadata_file,
        yolo_model_path = "finetuned.pt",
        clip_model_name = "openai/clip-vit-base-patch32",
        ocr_weight = 0.3,
        top_n_fitlayout = 3,
        top_n_yolo = 3,
        yolo_conf = 0.01,
        yolo_iou = 0.5,
        device = None
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.ocr_weight = ocr_weight
        self.top_n_fitlayout = top_n_fitlayout
        self.top_n_yolo = top_n_yolo
        self.yolo_conf = yolo_conf
        self.yolo_iou = yolo_iou

        print(f"[{self.__class__.__name__}] Loading models and reference database...")
        # load CLIP
        self.clip_model = CLIPVisionModel.from_pretrained(clip_model_name, local_files_only=True).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name, local_files_only=True)
        
        # load YOLO
        self.yolo_model = YOLO(yolo_model_path)

        # load easyOCR
        models_dir = "/app/models/easyocr"
        os.makedirs(models_dir, exist_ok=True)
        self.ocr_reader = easyocr.Reader(["en"], gpu=(self.device == "cuda"), verbose=False, model_storage_directory=models_dir, download_enabled=False)
        
        # setup FAISS index
        ref_matrix = np.load(db_vectors_file)
        if len(ref_matrix.shape) == 1: 
            ref_matrix = np.vstack(ref_matrix)
        ref_matrix = ref_matrix.astype(np.float32)
        with open(db_metadata_file, "r", encoding="utf-8") as f:
            self.ref_metadata_map = json.load(f)         
        self.index = faiss.IndexFlatIP(ref_matrix.shape[1]) 
        self.index.add(ref_matrix)


    def _get_embeddings(self, img_crop):
        """Get embedding representations of a cropped candidate logo"""
        embeddings = []

        try:
            img = img_crop.convert("RGBA")

            # determine whether to use black or white image background for transparent logos (or both, meaning 2 separate embeddings)
            alpha = img.split()[3]
            alpha_min, alpha_max = alpha.getextrema()
            is_opaque = (alpha_min == 255)
            use_white_bg = True
            use_black_bg = not is_opaque

            # no transparency, try to choose one background only
            if not is_opaque:
                arr = np.array(img)
                r, g, b, a = arr[:,:,0], arr[:,:,1], arr[:,:,2], arr[:,:,3]
                visible_mask = a > 50
                total_visible = np.sum(visible_mask)

                if total_visible > 0:
                    # ratio of near-black pixels in the crop
                    ratio_black = np.sum((r < 40) & (g < 40) & (b < 40) & visible_mask) / total_visible
                    # ratio of near-white pixels in the crop
                    ratio_white = np.sum((r > 215) & (g > 215) & (b > 215) & visible_mask) / total_visible

                    # use the opposite of the most present color
                    if ratio_black > 0.5: 
                        use_black_bg = False
                    if ratio_white > 0.5: 
                        use_white_bg = False
                    # fallback just in case
                    if not use_white_bg and not use_black_bg:
                        use_white_bg, use_black_bg = True, True

            # prepare background color configurations
            configs = []
            if use_white_bg: 
                configs.append((255, 255, 255, 255))
            if use_black_bg: 
                configs.append((0, 0, 0, 255))
            
            for bg_color in configs:
                # composite the logo on the background
                background = Image.new("RGBA", img.size, bg_color)
                comp = Image.alpha_composite(background, img).convert("RGB")
                comp_final = make_square_with_padding(comp, background_color=bg_color[:3])
                
                # ignore mostly uniform results
                stat_final = ImageStat.Stat(comp_final.convert("L"))
                if stat_final.stddev[0] < 5: 
                    continue
                
                # get the CLIP embedding
                inputs = self.clip_processor(images=comp_final, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                    emb = outputs.pooler_output.cpu().numpy()[0]
                # normalize embedding
                emb = emb / np.linalg.norm(emb)
                embeddings.append(emb.astype(np.float32))
                
        except Exception:
            pass
        return embeddings


    def _extract_candidates(self, image_path, img_obj, ttl_path):
        """Get crop candidates from image (from both YOLO and RDF graph)"""
        candidates = []
        w, h = img_obj.size
        
        # extract candidates from the RDF graph
        if ttl_path and os.path.exists(ttl_path):
            try:
                graph = Graph()
                graph.parse(ttl_path, format="turtle")

                # get all boxes
                box_nodes = graph.subjects(RDF.type, B.Box)
                fitlayout_cands = []

                for box in box_nodes:
                    # get box bounds (prefer visualBounds)
                    rect_node = graph.value(box, B.visualBounds)
                    if not rect_node:
                        rect_node = graph.value(box, B.bounds)
                    
                    # no bounds for this box, skip
                    if not rect_node:
                        continue

                    # get box coordinates and HTML tag
                    box_x = int(float(graph.value(rect_node, B.positionX) or 0))
                    box_y = int(float(graph.value(rect_node, B.positionY) or 0))
                    box_w = int(float(graph.value(rect_node, B.width) or 0))
                    box_h = int(float(graph.value(rect_node, B.height) or 0))
                    tag_node = graph.value(box, B.htmlTagName)
                    tag = str(tag_node).upper() if tag_node else "NONE"

                    # calculate heuristic score
                    score = calculate_box_score(box_x, box_y, box_w, box_h, h, w, tag=tag)
                    if score and score > 0:
                        fitlayout_cands.append({
                            "coords": (box_x, box_y, box_x + box_w, box_y + box_h),
                            "score": float(score),
                            "tag": tag,
                            "source": "FitLayout"
                        })
                
                # keep N best candidates
                fitlayout_cands = sorted(fitlayout_cands, key=lambda c: c["score"], reverse=True)[:self.top_n_fitlayout]
                candidates.extend(fitlayout_cands)
                
            except Exception as e:
                print(f"Error during RDF candidate extraction: {e}")

        # extract candidates with YOLO model
        try:
            # crop image to square
            crop_h = min(h, w)
            yolo_input = img_obj.crop((0, 0, w, crop_h)) if h > crop_h else img_obj

            # extraction
            results = self.yolo_model(yolo_input, conf=self.yolo_conf, iou=self.yolo_iou, imgsz=1280, rect=True, classes=[0], verbose=False)
            yolo_cands = []
            
            for box in results[0].boxes:
                # extract predicted bounding box coordinates and confidence
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                box_w, box_h = x2 - x1, y2 - y1
                conf = box.conf[0].item()
                
                # calculate heuristic score
                score = calculate_box_score(x1, y1, box_w, box_h, crop_h, w, tag="IMG")
                if score and score > 0:
                     yolo_cands.append({
                        "coords": (x1, y1, x2, y2),
                        "score": conf,
                        "tag": "YOLO_BOX",
                        "source": "YOLO"
                    })
            
            # keep N best candidates
            yolo_cands = sorted(yolo_cands, key=lambda c: c["score"], reverse=True)[:self.top_n_yolo]
            candidates.extend(yolo_cands)
            
        except Exception as e:
            print(f"Error during YOLO candidate extraction: {e}")
            
        return candidates


    def evaluate_single(self, url, image_path, ttl_path):
        """Classify a single screenshot"""

        result_dict = {
            "status": "success", "is_phishing": False,
            "culprit_brand": None, "culprit_sim": 0.0, "culprit_candidate": None
        }

        # missing screenshot
        if not os.path.exists(image_path):
            result_dict["status"] = "skipped"
            return result_dict

        try: 
            img_obj = Image.open(image_path)
        except Exception:
            result_dict["status"] = "skipped"
            return result_dict

        # extract logo candidates
        candidates = self._extract_candidates(image_path, img_obj, ttl_path)
        
        # possible OCR weights
        ocr_weights = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]

        # separate best match statistics for eeach OCR weight
        best_fusion_threat = {w: 0.0 for w in ocr_weights}
        max_vis_threat = {w: 0.0 for w in ocr_weights}
        txt_sim_threat = {w: 0.0 for w in ocr_weights}
        best_cand_threat = {w: None for w in ocr_weights}
        best_brand_threat = {w: None for w in ocr_weights}
        best_fusion_legit = {w: 0.0 for w in ocr_weights}
        max_vis_legit = {w: 0.0 for w in ocr_weights}

        for cand in candidates:
            x1, y1, x2, y2 = cand["coords"]
            crop = img_obj.crop((x1, y1, x2, y2))
            
            # run OCR on the candidate crop
            try:
                ocr_out = self.ocr_reader.readtext(np.array(crop.convert("RGB")), detail=0)
                cand_ocr_text = " ".join(ocr_out).strip().lower()
            except Exception:
                cand_ocr_text = ""

            # get candidate embeddings
            vecs = self._get_embeddings(crop)
            if not vecs:
                continue
            
            # evaluate all embeddings
            for q_vec in vecs:
                # search in the reference database
                D, I = self.index.search(q_vec.reshape(1, -1), k=self.index.ntotal)
                
                # go through all valid matches
                for k_idx in range(len(I[0])):
                    db_idx = int(I[0][k_idx])
                    if db_idx < 0: 
                        continue
                    
                    # visual similarity from the embedding index
                    vis_sim = float(D[0][k_idx])

                    # extract reference text and domains
                    db_entry = self.ref_metadata_map[db_idx]
                    ref_ocr = db_entry.get("ocr_text", "")
                    matched_domains = db_entry.get("domains",[])
                    
                    # calculate text similarity
                    current_txt_sim = 0.0
                    if len(ref_ocr) >= 3 and len(cand_ocr_text) >= 3:
                        current_txt_sim = get_text_similarity(cand_ocr_text, ref_ocr)
                    
                    # check if the matched logo belongs to the same domain
                    is_legit = check_domain_consistency(url, matched_domains)

                    # evaluate results with all OCR weights
                    for w in ocr_weights:
                        # fusion similarity is equal to visual similarity with a small OCR bonus
                        current_fusion_sim = vis_sim
                        if current_txt_sim > 0:
                            current_fusion_sim = min(1.0, vis_sim + (current_txt_sim * w))
                        
                        # domains match, update best legit match
                        if is_legit:
                            if current_fusion_sim > best_fusion_legit[w]:
                                best_fusion_legit[w] = current_fusion_sim
                                max_vis_legit[w] = vis_sim
                        # domains do not match, update best threat match
                        else:
                            if current_fusion_sim > best_fusion_threat[w]:
                                best_fusion_threat[w] = current_fusion_sim
                                max_vis_threat[w] = vis_sim
                                txt_sim_threat[w] = current_txt_sim
                                best_cand_threat[w] = cand
                                best_brand_threat[w] = matched_domains

        # pack results
        result_dict["weights_data"] = {}
        for w in ocr_weights:
            w_str = str(w)
            
            # remove PIL image before saving
            cand_clean = None
            if best_cand_threat[w]:
                cand_clean = best_cand_threat[w].copy()
                cand_clean.pop("crop", None)
            
            result_dict["weights_data"][w_str] = {
                "vis_sim_threat": max_vis_threat[w],
                "txt_sim_threat": txt_sim_threat[w],
                "vis_sim_legit": max_vis_legit[w],
                "culprit_sim": best_fusion_threat[w],
                "culprit_brand": list(best_brand_threat[w]) if best_brand_threat[w] else None,
                "culprit_candidate": cand_clean
            }
        
        # preserve top-level outputs for backwards compatibility
        default_w = self.ocr_weight
        result_dict["vis_sim_threat"] = max_vis_threat[default_w]
        result_dict["txt_sim_threat"] = txt_sim_threat[default_w]
        result_dict["vis_sim_legit"] = max_vis_legit[default_w]
        result_dict["culprit_sim"] = best_fusion_threat[default_w]
        result_dict["is_phishing"] = (best_fusion_threat[default_w] >= 0.85) and (best_fusion_threat[default_w] > best_fusion_legit[default_w])

        return result_dict
