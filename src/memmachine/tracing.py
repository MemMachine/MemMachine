# tracing.py
from jaeger_client import Config
import opentracing

_tracer = None

def init_tracer(service_name='backend', jaeger_host='jaeger', jaeger_port=6831):
    """Initialize and return the global tracer"""
    global _tracer
    if _tracer is not None:
        return _tracer
        
    config = Config(
        config={
            'sampler': {'type': 'const', 'param': 1},
            'logging': True,
            'local_agent': {
                'reporting_host': jaeger_host,
                'reporting_port': jaeger_port,
            },
        },
        service_name=service_name,
        validate=True
    )
    _tracer = config.initialize_tracer()
    return _tracer

def get_tracer():
    """Get the global tracer instance"""
    if _tracer is None:
        raise RuntimeError("Tracer not initialized. Call init_tracer first.")
    return _tracer