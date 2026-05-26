import asyncio
import time
import uuid
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from orchestrator import Orchestrator
from modules.utils import normalize_url

#web UI stuff
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="templates")


app = FastAPI(title="Phishing Detection API")

orchestrator = Orchestrator()

# max number of simultaneous scans
MAX_CONCURRENT_SCANS = int(os.getenv("MAX_CONCURRENT_SCANS", 4))
scan_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCANS)
MAX_QUEUE_SIZE = int(os.getenv("MAX_SCAN_QUEUE_SIZE", 20))

# task cache
cache_lock = asyncio.Lock()
active_tasks_cache = {}
CACHE_LIFETIME_SECONDS = 3600


class AnalyzeRequest(BaseModel):
    url: str
    manual: bool = False


# web UI endpoint
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# analysis endpoint
@app.post("/analyze")
async def analyze_url(request: AnalyzeRequest):
    # validate url
    try:
        target_url = await asyncio.to_thread(normalize_url, request.url)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    

    waiting = False
    wait_event = None

    # atomic cache block
    async with cache_lock:

        # delete tasks cached for too long
        current_time = time.time()
        keys_to_delete = [
            key for key, task_data in active_tasks_cache.items()
            if task_data.get("status") != "processing" and (current_time - task_data.get("timestamp", 0) > CACHE_LIFETIME_SECONDS)
        ]
        for key in keys_to_delete:
            active_tasks_cache.pop(key, None)

        # check cache
        cached_task = active_tasks_cache.get(target_url)

        # always wait if the same url is already being processed
        if cached_task and cached_task.get("status") == "processing":
            wait_event = cached_task.get("event")
            waiting = True
        
        # cache hit, return cached result (except for manual requests)
        elif cached_task and not request.manual:
            if cached_task.get("status") == "completed":
                return {**cached_task.get("result"), "cached": True}
            elif cached_task.get("status") == "error":
                raise HTTPException(status_code=500, detail=f"Previous scan of this URL failed: {cached_task.get('error_detail')}")

        # cache was not used, check number of waiting jobs
        if not waiting:
            waiters_count = len(scan_semaphore._waiters or [])
            if waiters_count >= MAX_QUEUE_SIZE:
                raise HTTPException(status_code=503, detail=f"Server is overloaded, try again later. Number of pending tasks: {waiters_count}")
            
            # create wait event for new tasks with the same url
            wait_event = asyncio.Event()
            active_tasks_cache[target_url] = {
                "status": "processing",
                "timestamp": time.time(),
                "event": wait_event
            }


    # wait until the analysis of current url is done
    if waiting:
        await wait_event.wait()

        async with cache_lock:
            final_cached = active_tasks_cache.get(target_url)

        if final_cached:
            if final_cached.get("status") == "completed":
                return {**final_cached.get("result"), "cached": True}
            
            raise HTTPException(status_code=500, detail=f"Previous cached scan of this URL failed: {final_cached.get('error_detail')}")
        
        raise HTTPException(status_code=500, detail=f"Error: cached result is missing")
    

    # run a new analysis
    task_id = str(uuid.uuid4())
    try:
        async with scan_semaphore:
            result = await asyncio.to_thread(orchestrator.run_analysis, target_url, task_id)

        # atomic cache update
        async with cache_lock:
            active_tasks_cache[target_url].update({
                "status": "completed",
                "result": result,
                "timestamp": time.time()
            })
        # wake up waiting requests
        wait_event.set()

        return result
    
    except asyncio.CancelledError:
        async with cache_lock:
            if active_tasks_cache.get(target_url, {}).get("status") == "processing":
                active_tasks_cache[target_url].update({
                    "status": "error",
                    "error_detail": "Task cancelled",
                    "timestamp": time.time()
                })
            wait_event.set()
        
        raise 
    
    except Exception as e:
        async with cache_lock:
            if active_tasks_cache.get(target_url, {}).get("status") == "processing":
                active_tasks_cache[target_url].update({
                    "status": "error",
                    "error_detail": str(e),
                    "timestamp": time.time()
                })
            wait_event.set()
        
        raise HTTPException(status_code=500, detail=str(e))
    

# status endpoint
@app.get("/status")
async def status_check():
    return {"status": "ok"}