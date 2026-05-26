import os
import shutil
import subprocess
import requests
import base64
import json
import joblib
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor
from subprocess import CalledProcessError, TimeoutExpired

from modules.module_favicon import FaviconClassifier
from modules.module_logo import LogoClassifier
from modules.module_cnn import CNNClassifier
from modules.module_gnn import GNNClassifier
from modules.utils import normalize_url

# scan result directory
LIVE_SCANS_DIR = "live_scans_temp"

# favicon reference database
FAVICON_REF_DB_PATH = "models/favicon_ref_db.json"

# logo reference database, metadata and detection model
LOGO_REF_DB_PATH = "models/logo_reference_embeddings.npy"
LOGO_REF_METADATA_PATH = "models/logo_reference_metadata.json"
LOGO_MODEL_PATH = "models/yolo_finetuned.pt"

# cnn model
CNN_MODEL_PATH = "models/cnn_model.pth"

# gnn model
GNN_MODEL_PATH = "models/gnn_model.pt"

# ensemble model, roc thresholds
ENSEMBLE_MODEL_PATH = "models/meta_model.joblib"
MODEL_THRESHOLDS = "models/thresholds.json"


class TaskLogger:
    def __init__(self, task_id, url):
        self.prefix = self.get_log_prefix(task_id, url)

    def get_log_prefix(self, task_id, url): 
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")

        if len(domain) > 15:
            domain = domain[:12] + "..."
        return f"[{domain:^15}]"
        
    def info(self, msg):
        print(f"{self.prefix} {msg}")

    def warn(self, msg):
        print(f"{self.prefix}\033[33m WARNING: \033[0m{msg}")
        
    def error(self, msg):
        print(f"{self.prefix}\033[91m ERROR: \033[0m{msg}")


class Orchestrator:
    def __init__(self):
        os.makedirs(LIVE_SCANS_DIR, exist_ok=True)

        # initialize models
        print("[Orchestrator] Initializing models...")
        with open(FAVICON_REF_DB_PATH, "r", encoding="utf-8") as f:
            fav_db = {k: set(v) for k, v in json.load(f).items()}
        
        self.favicon_model = FaviconClassifier(ref_db=fav_db, threshold=9)
        self.logo_model = LogoClassifier(db_vectors_file=LOGO_REF_DB_PATH, db_metadata_file=LOGO_REF_METADATA_PATH, yolo_model_path=LOGO_MODEL_PATH)
        self.cnn_model = CNNClassifier(model_path=CNN_MODEL_PATH)
        self.gnn_model = GNNClassifier(model_path=GNN_MODEL_PATH, num_node_features=52)
        self.meta_model = joblib.load(ENSEMBLE_MODEL_PATH)

        with open(MODEL_THRESHOLDS, "r", encoding="utf-8") as f:
            self.thresholds = json.load(f)

        print("[Orchestrator] Ready.")


    def _fetch_favicon(self, url, task_dir):
        """Get favicon from a given URL adress"""

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        fav_path = os.path.join(task_dir, "favicon.png")
        
        # manual search
        try:
            html_resp = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(html_resp.content, "html.parser")
            icon_link = soup.find("link", rel=lambda r: r and "icon" in r.lower())
            
            if icon_link and icon_link.get("href"):
                href = icon_link["href"]
                if href.startswith("data:image"):
                    header, encoded = href.split(",", 1)
                    with open(fav_path, "wb") as f:
                        f.write(base64.b64decode(encoded))
                    return fav_path
                else:
                    icon_url = urljoin(url, href)
                    img_resp = requests.get(icon_url, headers=headers, timeout=5)
                    if img_resp.status_code == 200:
                        with open(fav_path, "wb") as f:
                            f.write(img_resp.content)
                        return fav_path
        except:
            pass
                        
        # check for /favicon.ico
        try:
            ico_resp = requests.get(urljoin(url, "/favicon.ico"), headers=headers, timeout=5)
            if ico_resp.status_code == 200:
                with open(fav_path, "wb") as f:
                    f.write(ico_resp.content)
                return fav_path
        except:
            pass


        # try google favicon api
        try:
            domain = url.split("//")[-1].split("/")[0]
            api_url = f"https://www.google.com/s2/favicons?domain={domain}&sz=128"
            r = requests.get(api_url, headers=headers, timeout=5)
            
            if r.status_code == 200 and r.headers.get("Content-Type", "").startswith("image"):
                with open(fav_path, "wb") as f:
                    f.write(r.content)
                return fav_path
        except:
            pass

        return None


    def run_analysis(self, url, task_id):
        log = TaskLogger(task_id, url)
        task_dir = os.path.join(LIVE_SCANS_DIR, task_id)
        os.makedirs(task_dir, exist_ok=True)
        
        try:
            log.info("1/3 Starting task")
            
            # get favicon
            fav_path = self._fetch_favicon(url, task_dir)
            if not fav_path:
                log.warn("Favicon fetch failed")
            
            # run fitlayout
            raw_ttl_path = os.path.join(task_dir, "site_raw.ttl")
            final_ttl_path = os.path.join(task_dir, "site.ttl")
            screenshot_path = os.path.join(task_dir, "screenshot.png")
            puppeteer_json_output_path = os.path.join(task_dir, "puppeteer_output.json")

            # new subprocess environment
            run_env = os.environ.copy()
            # set output dir env variable
            run_env["TASK_OUT_DIR"] = os.path.join("/app", task_dir)
            run_env["TASK_OUT_DIR"] = run_env["TASK_OUT_DIR"].replace("\\", "/")
            
            cmd = [
                "./fitlayout.sh",
                "RENDER",
                "-b", "puppeteer",
                "--options=width=1920,height=1080,persist=1",
                url,
                "EXPORT",
                "-f", "png",
                "-o", f"screenshot.png",
                "EXPORT",
                "-f", "turtle",
                "-o", f"site_raw.ttl"
            ]
            
            try:
                log.info("2/3 Running FitLayout")
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60, env=run_env)
            except TimeoutExpired:
                log.error("FitLayout time limit has expired")
                raise Exception("Time limit for FitLayout has expired.")
            except CalledProcessError as e:
                log.error(f"FitLayout failed: {e.stderr.decode('utf-8') if e.stderr else 'Unknown error'}")
                raise Exception(f"FitLayout failed.")
            
            # extract final redirected url from fitlayout output json
            fitlayout_rendered_url = url 
            if os.path.exists(puppeteer_json_output_path):
                try:
                    with open(puppeteer_json_output_path, "r", encoding="utf-8") as f:
                        fitlayout_data = json.load(f)
                    
                    if fitlayout_data and "page" in fitlayout_data and "url" in fitlayout_data["page"]:
                        fitlayout_rendered_url = fitlayout_data["page"]["url"]
                        fitlayout_rendered_url = normalize_url(fitlayout_rendered_url)
                    else:
                        log.warn("'page.url' attribute was not found")
                except Exception as e:
                    log.warn(f"Puppeteer json processing failed: {e}")
            else:
                log.warn(f"File '{puppeteer_json_output_path}' was not found")
            target_url = fitlayout_rendered_url


            # clean TTL file
            if not os.path.exists(raw_ttl_path):
                log.error(f"TTL file '{raw_ttl_path}' was not found")
                raise Exception(f"TTL file was not found at {raw_ttl_path}!")
            try:
                with open(raw_ttl_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                new_lines = []
                for line in lines:
                    if "b:pngImage" in line:
                        # fix the previous line to terminate the turtle statement correctly
                        if new_lines:
                            prev_line = new_lines[-1].rstrip()
                            if prev_line.endswith(";"):
                                new_lines[-1] = prev_line[:-1] + " .\n"
                    else:
                        new_lines.append(line)

                # write the cleaned TTL
                with open(final_ttl_path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)

                os.remove(raw_ttl_path)            
            except Exception as e:
                log.error(f"TTL processing failed: {e}")
                raise Exception(f"TTL processing error: {e}")
            
            if not os.path.exists(screenshot_path):
                log.warn(f"Screenshot extraction failed")


            # analysis
            log.info("3/3 Analyzing extracted data")
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_fav = executor.submit(self.favicon_model.evaluate_single, fav_path, target_url) if fav_path else None
                future_logo = executor.submit(self.logo_model.evaluate_single, target_url, screenshot_path, final_ttl_path)
                future_cnn = executor.submit(self.cnn_model.evaluate_single, screenshot_path)
                future_gnn = executor.submit(self.gnn_model.evaluate_single, final_ttl_path)
                
                # wait for results
                fav_res = future_fav.result() if future_fav else {}
                logo_res = future_logo.result()
                cnn_res = future_cnn.result()
                gnn_res = future_gnn.result()
                

            # favicon module returns threat and legit similarity
            if fav_res.get("status", "") == "success":
                fav_sim_threat = fav_res.get("sim_threat", 0)
                fav_sim_legit = fav_res.get("sim_legit", 0)
            else:
                fav_sim_threat, fav_sim_legit = 0, 0
            fav_fusion = 0 if fav_sim_legit > fav_sim_threat else fav_sim_threat
            
            # logo module returns threat (visual/text) and legit (visual) similarities
            if logo_res.get("status") == "success":
                logo_res_weight = logo_res.get("weights_data").get("0.3")
                logo_vis_threat = logo_res_weight.get("vis_sim_threat", 0.0)
                logo_txt_threat = logo_res_weight.get("txt_sim_threat", 0.0)
                logo_vis_legit = logo_res_weight.get("vis_sim_legit", 0.0)
                logo_fusion = logo_res_weight.get("culprit_sim", 0.0)
                logo_brand = logo_res_weight.get("culprit_brand", None)
            else:
                logo_vis_threat, logo_txt_threat, logo_vis_legit, logo_fusion = 0.0, 0.0, 0.0, 0.0
            
            # cnn module returns probability
            cnn_prob = cnn_res.get("probability", 0.0) if cnn_res.get("status") == "success" else 0.0
            
            # gnn module returns probability
            gnn_prob = gnn_res.get("probability", 0.0) if gnn_res.get("status") == "success" else 0.0
            
            # feature vector for ensemble 
            features = [[fav_sim_threat, fav_sim_legit, logo_vis_threat, logo_txt_threat, logo_vis_legit, cnn_prob, gnn_prob]]
            
            # get final decision
            risk_score = float(self.meta_model.predict_proba(features)[0][1])
  
            # if classified as phishing, check for specific reasons
            reasons = []
            matched_target = None
            if risk_score >= 0.5:
                if logo_fusion >= self.thresholds.get("logo_threshold", 0.85):
                    reasons.append("logo_identity_mismatch")
                    if logo_res and logo_brand:
                        matched_target = list(logo_brand)[0]
                        
                if fav_fusion >= self.thresholds.get("favicon_threshold", 0.85):
                    reasons.append("favicon_identity_mismatch")
                    if not matched_target and fav_res and fav_res.get("nearest_threat_domains"):
                        matched_target = list(fav_res["nearest_threat_domains"])[0]
                        
                if gnn_prob >= self.thresholds.get("gnn_threshold", 0.5):
                    reasons.append("structural_anomaly")
                    
                if cnn_prob >= self.thresholds.get("cnn_threshold", 0.5):
                    reasons.append("visual_anomaly")
                    
                # fallback - if there are no reasons, ensemble must have made the decision
                if not reasons:
                    reasons.append("ensemble_suspicion")
                    
            # return the result
            result = {
                "verdict": "phishing" if risk_score >= 0.5 else "legitimate",
                "risk_score": round(risk_score, 4)
            }
            if risk_score >= 0.5:
                result["reasons"] = reasons
            if matched_target is not None:
                result["matched_target"] = matched_target
            
            log.info(result)
            return result

        finally:
            # clean up
            try:
                shutil.rmtree(task_dir)
            except Exception as e:
                log.warn(f"Failed to delete task directory {task_dir}: {e}")