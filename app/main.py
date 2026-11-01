from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router
from app.core.config import get_settings
from app.db import init_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    get_settings().upload_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Civic Triage API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=get_settings().allowed_origins,
                   allow_credentials=False, allow_methods=["GET", "POST", "PATCH"], allow_headers=["*"])
app.include_router(router)

@app.get("/")
def start():
    return {"hello I am meow"}

@app.get("/api/health")
def health():
    return {"status": "ok"}
