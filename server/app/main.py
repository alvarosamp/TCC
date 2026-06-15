from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from httpx import get
from server.app.routes import ota,events,feedback, devices
from server.app.metrics import render_prometheus_metrics

app = FastAPI(
    title = 'TCC API',
    description ='Servidor para o ota',
    version = '1.0.0',
    redirect_slashes=False,
)

app.include_router(ota.router, prefix='/ota', tags=['OTA'])
app.include_router(events.router, prefix='/events', tags=['Events'])
app.include_router(feedback.router, prefix='/feedback', tags=['Feedback'])
app.include_router(devices.router, prefix='/devices', tags=['Devices'])

@app.get("/")
def root():
    return {'status': 'online',
            'message': 'TCC TinyML OTA Feedback Server',
            'version': '1.0.0'}
    
@app.get('/health')
def health_check():
    return {'status': 'healthy'}

@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics():
    return render_prometheus_metrics()
