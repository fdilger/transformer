from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active connections and latest metrics
active_connections = set()
latest_metrics = None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    logger.info(f"Client connected. Total connections: {len(active_connections)}")
    
    try:
        while True:
            # Receive message from the client
            data = await websocket.receive_json()
            #logger.info(f"Received metrics update")
            
            # Store the latest metrics
            global latest_metrics
            latest_metrics = data
            
            # Broadcast to all other clients
            await broadcast_metrics(data)
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(active_connections)}")
    except Exception as e:
        logger.error(f"Error in websocket connection: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

async def broadcast_metrics(metrics: dict):
    """Send metrics to all connected clients except the sender"""
    for connection in active_connections.copy():
        try:
            await connection.send_json(metrics)
        except Exception as e:
            logger.error(f"Error broadcasting metrics: {e}")
            active_connections.remove(connection)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
