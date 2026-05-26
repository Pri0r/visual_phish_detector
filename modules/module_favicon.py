import os
import numpy as np
import imagehash
import faiss
from io import BytesIO
from PIL import Image, ImageStat

from utils import get_domain, check_domain_consistency

# ignore DecompressionBombError
Image.MAX_IMAGE_PIXELS = None

def hex_to_uint8_array(hex_str):
    """Converts hexadecimal hash to numpy array"""
    try:
        h = int(hex_str, 16)
        byte_array = h.to_bytes(8, "big")
        return np.frombuffer(byte_array, dtype=np.uint8)
    except Exception:
        return np.zeros(8, dtype=np.uint8)

def get_phash(image_input):
    """Loads image, resizes it to 64x64, places it on a black/white background and calculates perceptual hash"""
    try:
        # load image
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                return None
            img = Image.open(image_input)
        elif isinstance(image_input, (bytes, bytearray)):
            img = Image.open(BytesIO(image_input))
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            return None
        
        # ignore extremely small images
        if img.width < 4 or img.height < 4:
            return None
            
        # convert and resize
        img = img.convert("RGBA")
        img = img.resize((64, 64), Image.Resampling.LANCZOS)
        
        # analyse average brightness to determine which background to use
        r, g, b, alpha = img.split()
        grayscale = img.convert("L")
        stat = ImageStat.Stat(grayscale, mask=alpha)
        
        # ignore fully transparent images
        if stat.count[0] == 0:
            return None
            
        # place the image on the background
        avg_brightness = stat.mean[0]
        bg_color = (0, 0, 0, 255) if avg_brightness > 128 else (255, 255, 255, 255)
        background = Image.new("RGBA", img.size, bg_color)
        combined = Image.alpha_composite(background, img)

        # ignore images consisting of a single color only (e.g. a black rectangle)
        final_gray = combined.convert("L")
        stat_final = ImageStat.Stat(final_gray)
        std_dev = stat_final.stddev[0]
        if std_dev < 1:
            return None
        
        return str(imagehash.phash(combined.convert("L"), hash_size=8))
        
    except Exception:
        return None


class FaviconClassifier:
    """Favicon phishing classifier"""
    def __init__(self, ref_db, threshold = 8, max_domains_per_hash = 100):
        self.threshold = threshold
        self.ref_domains_map = []
        ref_vectors = []
        
        print(f"[{self.__class__.__name__}] Loading reference database...")
        # setup FAISS index
        for hash_str, domains in ref_db.items():
            # ignore hashes that are too simple or hashes that have too many domains associated with them
            if hash_str.startswith("00000000") or len(domains) > max_domains_per_hash:
                continue
            ref_vectors.append(hex_to_uint8_array(hash_str))
            self.ref_domains_map.append(domains)
        ref_matrix = np.array(ref_vectors)
        self.index = faiss.IndexBinaryFlat(64) 
        self.index.add(ref_matrix)

    def evaluate_single(self, favicon_path, current_domain):
        """Classify a single favicon"""
        # missing domain
        if not current_domain:
            return {"status": "skipped", "is_phishing": False}
        
        current_domain = get_domain(current_domain)
        phash = get_phash(favicon_path)
        # invalid favicon
        if not phash:
            return {"status": "skipped", "is_phishing": False}

        # FAISS lookup
        query_vec = hex_to_uint8_array(phash).reshape(1, -1)
        lims, D, I = self.index.range_search(query_vec, 64) 
        
        min_dist_threat = 64
        min_dist_legit = 64
        nearest_threat_domains = []

        # is there at least one match?
        if lims[1] > 0:
            # go through all matches
            for k in range(lims[0], lims[1]):
                dist = int(D[k])
                db_idx = I[k]
                
                # make sure the index is valid
                if db_idx < len(self.ref_domains_map):
                    matched_domains = self.ref_domains_map[db_idx]
                    
                    # check if the matching reference sample comes from the same domain
                    if check_domain_consistency(current_domain, matched_domains):
                        # same domain, update min legit distance
                        if dist < min_dist_legit:
                            min_dist_legit = dist
                    else:
                        # different domain, update min threat distance
                        if dist < min_dist_threat:
                            min_dist_threat = dist
                            nearest_threat_domains = list(matched_domains)

        # classified as phishing if under the threshold and also closer than the min legit distance
        predicted_phishing = (min_dist_threat <= self.threshold) and (min_dist_threat < min_dist_legit)

        # convert distance to similarity for consistency with logo module
        sim_threat = max(0, 64 - min_dist_threat) / 64.0
        sim_legit = max(0, 64 - min_dist_legit) / 64.0

        return {
            "status": "success",
            "is_phishing": predicted_phishing,
            "sim_threat": sim_threat,
            "sim_legit": sim_legit,
            "nearest_threat_domains": nearest_threat_domains,
            "phash": phash
        }
