from src.backend_telemetry_model import BackendTelemetryScorer


scorer = BackendTelemetryScorer("models_saved/backend_telemetry_isoforest.joblib")


def handle_backend_request(request: dict) -> dict:
    return scorer.handle_packet(
        satellite_id=request["satellite_id"],
        telemetry={
            "battery_voltage": request["battery_voltage"],
            "temperature": request["temperature"],
            "cpu_usage": request["cpu_usage"],
            "signal_strength": request["signal_strength"],
        },
    )


if __name__ == "__main__":
    sample_request = {
        "satellite_id": "SAT-001",
        "battery_voltage": 12.4,
        "temperature": 38.2,
        "cpu_usage": 0.43,
        "signal_strength": 0.91,
    }

    print(handle_backend_request(sample_request))