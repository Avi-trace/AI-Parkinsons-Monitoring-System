import unittest

from backend.app import app


class SpiralRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_spiral_health_route_is_registered(self):
        response = self.client.get("/api/spiral-test/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_spiral_analysis_route_accepts_valid_payload(self):
        points = []
        for i in range(30):
            angle = i / 10.0
            radius = 10 + i * 2
            points.append({
                "x": 320 + radius * 0.8 * (1 if i % 2 == 0 else -1),
                "y": 160 + radius * 0.6 * (1 if i % 3 == 0 else -1),
                "t": 1000 + i * 20,
            })

        response = self.client.post(
            "/api/spiral-test",
            json={"patient_id": "PT001", "hand": "dominant", "points": points},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["patient_id"], "PT001")
        self.assertGreaterEqual(payload["point_count"], 30)


if __name__ == "__main__":
    unittest.main()
