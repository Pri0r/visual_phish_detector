import torch
import tldextract
import numpy as np
from torch_geometric.data import Data
from rdflib import Graph, Namespace
from rdflib.namespace import RDF, RDFS
from urllib.parse import urlparse
import ipaddress
import socket

# rdf namespace
B = Namespace("http://fitlayout.github.io/ontology/render.owl#")

SUSPICIOUS_KEYWORDS = [
    "login", "sign in", "password", "username", "email", "verify", 
    "account", "secure", "bank", "update", "confirm"
]

TAG_LIST = [
    "div", "a", "p", "img", "span", "header", "footer", "nav", "ul", "li",
    "h1", "h2", "h3", "input", "button", "form", "select", "body", "i", "none"
]

DISPLAY_TYPES = [
    "block", "inline", "inline-block", "list-item", "flex", "grid", "table", "none"
]


def normalize_url(url):
    """Normalize and validate URL"""

    url = url.strip().rstrip("/")
    if not url:
        raise ValueError("URL cannot be empty!")
        
    # add schema if missing
    if not url.lower().startswith(("http://", "https://")):
        url = "http://" + url
        
    # check if valid format
    parsed = urlparse(url)
    if not parsed.netloc:
        raise ValueError("Invalid URL format!")
    
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Invalid hostname!")
    try:
        # dns resolve
        ip = socket.gethostbyname(hostname)
        ip_obj = ipaddress.ip_address(ip)

        # block internal ips
        if (
            ip_obj.is_private or
            ip_obj.is_loopback or
            ip_obj.is_reserved or
            ip_obj.is_link_local
        ):
            raise ValueError("Access to internal networks is not allowed!")

    except Exception:
        raise ValueError("Invalid or unresolvable domain!")
        
    return url


def get_domain(url):
    """Gets domain + TLD from URL. Example: 'http://site.abcd.com' -> 'abcd.com'"""
    if not url:
        return None
    
    try:
        url = str(url)
        # remove image file extensions if present
        if url.endswith(".png") or url.endswith(".jpg") or url.endswith(".ico"):
            url = url.rsplit(".", 1)[0]

        extracted = tldextract.extract(url)
        # standard domain + suffix
        if extracted.domain and extracted.suffix:
            return f"{extracted.domain}.{extracted.suffix}".lower()
         # ip address or hostname without suffix
        elif extracted.domain and not extracted.suffix:
            return (extracted.domain).lower()
        return None
    except Exception: 
        return None
    

def is_trusted_tld(tld_suffix):
    """Checks if TLD can be trusted (ccTLDs or very common ones)"""
    if not tld_suffix:
        return False
    
    parts = tld_suffix.lower().split(".")

    # single part TLDs
    if len(parts) == 1:
        tld = parts[0]
        if tld in {"com", "net", "org", "edu", "gov", "mil", "int"}:
            return True
        # ccTLD are two letters, ignore the ones that are often exploited
        if len(tld) == 2 and tld not in {"io", "tk", "ml", "ga", "cf", "gq"}:
            return True
        return False
    
    # two part TLDs
    if len(parts) == 2:
        p1, p2 = parts[0], parts[1]
        
        # allow two part country code TLDs except for the ones that are often exploited
        if len(p2) == 2 and p2 not in {"tk", "ml", "ga", "ge", "cf", "gq", "py", "gl", "io", "ly", "bn"}:
            allowed_p1 = {"com", "co", "org", "net", "edu", "ac", "gov", "gob", "gouv", "govt", "go"}
            if p1 in allowed_p1:
                return True
                
        return False

    return False


def check_domain_consistency(current, domains_list):
    """Checks if a domain is consistent with a set of reference domains"""
    try:
        curr_ext = tldextract.extract(str(current))
        curr_root, curr_suffix = curr_ext.domain.lower(), curr_ext.suffix.lower()
        curr_full = f"{curr_root}.{curr_suffix}"     
    except ValueError:
        return False
    
    for domain in domains_list:
        db_ext = tldextract.extract(str(domain))
        db_root, db_suffix = db_ext.domain.lower(), db_ext.suffix.lower()
        db_full = f"{db_root}.{db_suffix}"

        # domains are the same, identity is consistent
        if curr_full == db_full:
            return True
            
        # if the SLD matches but the TLD doesnt, other trusted TLDs can also be accepted
        if curr_root == db_root and is_trusted_tld(curr_suffix):
            return True
            
    return False


class EarlyStopping:
    """Halts training after specified number of non-improved iterations to avoid overfitting"""
    def __init__(self, patience = 10, verbose = False, delta = 0.0, path = "checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        
    def __call__(self, val_loss, model):
        """Called after every train epoch to check whether to stop"""
        score = -val_loss
        # first recorded score, save checkpoint
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        # no improvement, increase counter and stop if finished
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        # score improved, re-save checkpoint
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model checkpoint"""
        if self.verbose:
            print(f"Saving checkpoint with decreased validation loss: {self.val_loss_min:.6f} -> {val_loss:.6f}")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def hex_to_normalized_rgb(hex_color):
    """Converts hexadecimal color string to normalized RGB values"""
    if not hex_color or hex_color == "none":
        return [0.0, 0.0, 0.0]
    hex_color = hex_color.lstrip("#")

    if len(hex_color) == 3:
        hex_color = "".join([c*2 for c in hex_color])

    try:
        return [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    except: 
        return [0.0, 0.0, 0.0]


def rdf_to_pyg_data(file_path, label):
    """Parses RDF graph into pytorch geometric object"""

    # load RDF graph
    graph = Graph()
    graph.parse(file_path, format="turtle")

    # get page dimensions
    page_node = next(graph.subjects(RDF.type, B.Page), None)
    page_w, page_h = 0, 0
    if page_node:
        page_w = int(graph.value(page_node, B.width) or 0)
        page_h = int(graph.value(page_node, B.height) or 0)

    # load and sort rdf boxes
    box_nodes = list(graph.subjects(RDF.type, B.Box))
    def get_order(node):
        o = graph.value(node, B.documentOrder)
        return int(o) if o is not None else 0
    box_nodes.sort(key=get_order)

    # map RDF node URIs to graph node indexes
    node_uri_to_idx = {node: i for i, node in enumerate(box_nodes)}

    box_data = []
    max_x_w, max_y_h = 0, 0

    for box in box_nodes:
        # prefer visual bounds
        rect_node = graph.value(box, B.visualBounds)
        if not rect_node:
            rect_node = graph.value(box, B.bounds)
        
        # box coordinates
        x = float(graph.value(rect_node, B.positionX) or 0)
        y = float(graph.value(rect_node, B.positionY) or 0)
        w = float(graph.value(rect_node, B.width) or 0)
        h = float(graph.value(rect_node, B.height) or 0)

        # b:Page could be size 0, save max size just to be sure
        max_x_w = max(max_x_w, x + w)
        max_y_h = max(max_y_h, y + h)

        # extract box info
        tag = str(graph.value(box, B.htmlTagName) or "none").lower()
        display = str(graph.value(box, B.displayType) or "none").lower()
        visible = str(graph.value(box, B.visible) or "true").lower() == "true"
        bg_sep = str(graph.value(box, B.backgroundSeparated) or "false").lower() == "true"
        text = str(graph.value(box, B.text) or "").lower()
        bg_color = str(graph.value(box, B.backgroundColor))
        fg_color = str(graph.value(box, B.color))
        font_size = float(graph.value(box, B.fontSize) or 0)
        font_weight = float(graph.value(box, B.fontWeight) or 0)
        content_len = float(graph.value(box, B.contentLength) or 0)
        font_style = float(graph.value(box, B.fontStyle) or 0.0)
        underline = float(graph.value(box, B.underline) or 0.0)

        # extract attributes
        attrs = {}
        for attr_node in graph.objects(box, B.hasAttribute):
            attr_name = str(graph.value(attr_node, RDFS.label) or "").lower()
            attr_val = str(graph.value(attr_node, RDF.value) or "").lower()
            attrs[attr_name] = attr_val

        # merge all box data
        box_data.append({
            "x": x, "y": y, "w": w, "h": h,
            "tag": tag, "display": display, "visible": visible, "bg_sep": bg_sep,
            "text": text, "bg_color": bg_color, "fg_color": fg_color,
            "font_size": font_size, "font_weight": font_weight,
            "font_style": font_style, "underline": underline,
            "content_len": content_len, "attrs": attrs
        })

    # fix zero page size if needed
    if page_w <= 0:
        page_w = max(max_x_w, 1920)
    if page_h <= 0:
        page_h = max(max_y_h, 1080)

    # create final feature vector
    node_features = []
    for bd in box_data:
        # normalize values to (0, 1.0)
        pos_x = max(0.0, min(bd["x"] / page_w, 1.0))
        pos_y = max(0.0, min(bd["y"] / page_h, 1.0))
        width = max(0.0, min(bd["w"] / page_w, 1.0))
        height = max(0.0, min(bd["h"] / page_h, 1.0))

        is_visible = 1.0 if bd["visible"] else 0.0
        is_bg_separated = 1.0 if bd["bg_sep"] else 0.0
        kw_count = sum(bd["text"].count(kw) for kw in SUSPICIOUS_KEYWORDS)
        kw_score = min(kw_count / 5.0, 1.0) 
        c_len = min(bd["content_len"] / 1000.0, 1.0)
        f_size = min(bd["font_size"] / 100.0, 1.0)
        font_style = bd["font_style"]   
        underline = bd["underline"]

        # dead href links
        is_dead_link = 0.0
        if bd["tag"] == "a":
            href = bd["attrs"].get("href", "").strip().lower()
            if not href or href.startswith("#") or href.startswith("javascript"):
                is_dead_link = 1.0

        # actions leading to .php
        has_php_action = 0.0
        if bd["tag"] == "form":
            action = bd["attrs"].get("action", "").strip().lower()
            if ".php" in action:
                has_php_action = 1.0

        # password and text inputs
        is_password = 1.0 if bd["attrs"].get("type") == "password" else 0.0
        is_text_input = 1.0 if bd["attrs"].get("type") in ["text", "email"] else 0.0

        # html tags, display types as one-hot vectors
        tag_vec = [0.0] * (len(TAG_LIST) + 1)
        try: 
            tag_vec[TAG_LIST.index(bd["tag"])] = 1.0
        except ValueError:
            tag_vec[-1] = 1.0
        disp_vec = [0.0] * (len(DISPLAY_TYPES) + 1)
        try: 
            disp_vec[DISPLAY_TYPES.index(bd["display"])] = 1.0
        except ValueError:
            disp_vec[-1] = 1.0

        # colors
        bg_c = hex_to_normalized_rgb(bd["bg_color"])
        fg_c = hex_to_normalized_rgb(bd["fg_color"])

        feats = [
            pos_x, pos_y, width, height,
            f_size, bd["font_weight"], font_style, underline, c_len,
            is_visible, is_bg_separated, kw_score,
            is_dead_link, has_php_action, is_password, is_text_input
        ] + tag_vec + disp_vec + bg_c + fg_c

        node_features.append(feats)

    # graph edges
    edge_list = []
    for box in box_nodes:
        parent = graph.value(box, B.isChildOf)
        if parent:
            c_idx = node_uri_to_idx.get(box)
            p_idx = node_uri_to_idx.get(parent)
            if c_idx is not None and p_idx is not None:
                edge_list.append([c_idx, p_idx])

    # convert to pytorch tensor
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2,0), dtype=torch.long)
    y = torch.tensor([label], dtype=torch.float)
    
    pyg_data = Data(x=x, edge_index=edge_index, y=y)
    pyg_data.page_height = torch.tensor([page_h], dtype=torch.float)
    return pyg_data
