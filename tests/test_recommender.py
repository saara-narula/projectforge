import unittest
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import app
from recommender import ProjectRecommender

class ProjectRecommenderTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.client = self.app.test_client()
        self.recommender = ProjectRecommender()

    def test_projects_loaded(self):
        projects = self.recommender.get_all_projects()
        self.assertGreater(len(projects), 10)

    def test_collaborative_matrices_built(self):
        # Verify CF matrices are populated
        self.assertGreater(len(self.recommender.user_ratings), 0)
        self.assertGreater(len(self.recommender.item_ratings), 0)
        
        # Test item-item similarity
        test_id = "medical-imaging-diagnostic-cad"
        similar_items = self.recommender.get_collaborative_similar_items(test_id, top_k=2)
        self.assertIsInstance(similar_items, list)

    def test_hybrid_modes(self):
        # Test Hybrid vs Content vs Collaborative modes
        recs_hybrid = self.recommender.recommend(
            academic_background="Computer Science",
            domains=["Artificial Intelligence"],
            tech_stack=["Python", "PyTorch"],
            algorithm_mode="hybrid",
            top_n=3
        )
        recs_content = self.recommender.recommend(
            academic_background="Computer Science",
            domains=["Artificial Intelligence"],
            tech_stack=["Python", "PyTorch"],
            algorithm_mode="content",
            top_n=3
        )
        recs_collab = self.recommender.recommend(
            academic_background="Computer Science",
            domains=["Artificial Intelligence"],
            tech_stack=["Python", "PyTorch"],
            algorithm_mode="collaborative",
            top_n=3
        )
        self.assertEqual(len(recs_hybrid), 3)
        self.assertEqual(len(recs_content), 3)
        self.assertEqual(len(recs_collab), 3)

    def test_rate_api(self):
        # Test submitting a live peer rating
        response = self.client.post("/api/rate",
                                   data=json.dumps({
                                       "project_id": "medical-imaging-diagnostic-cad",
                                       "rating": 5,
                                       "major": "Computer Science"
                                   }),
                                   content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("updated_stats", data)

    def test_index_route(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"ProjectForge", response.data)
        self.assertIn(b"collaborative", response.data.lower())
        self.assertIn(b"content-based", response.data.lower())
        self.assertIn(b"hybrid", response.data.lower())

    def test_default_recommend_hybrid_scoring(self):
        # Default recommend() without specifying algorithm_mode should run in hybrid mode
        recs = self.recommender.recommend(
            academic_background="Computer Science",
            domains=["Artificial Intelligence"],
            tech_stack=["Python"]
        )
        self.assertGreater(len(recs), 0)
        first = recs[0]
        self.assertIn("content_score", first)
        self.assertIn("collaborative_score", first)
        self.assertGreater(first["content_score"], 0)
        self.assertGreater(first["collaborative_score"], 0)

    def test_project_detail_collab_insights(self):
        projects = self.recommender.get_all_projects()
        test_id = projects[0]["id"]
        response = self.client.get(f"/project/{test_id}")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Collaborative Filtering", response.data)

if __name__ == "__main__":
    unittest.main()
