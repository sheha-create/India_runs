"""
tests/test_components.py — Unit tests verifying components of the ranking system.
Uses a mock LLM client to run validation offline and fast.
"""
import unittest
from unittest.mock import MagicMock

from schemas.role_profile import RoleProfile
from schemas.candidate_profile import CandidateProfile
from modules.jd_parser import JDParser
from modules.candidate_profiler import CandidateProfiler
from modules.embedder import Embedder
from modules.ranker import Ranker


class MockLLMClient:
    def __init__(self, response_dict):
        self.response = response_dict

    def call(self, prompt: str) -> dict:
        return self.response


class TestCandidateRankingSystem(unittest.TestCase):

    def test_role_profile_embedding_text(self):
        profile = RoleProfile(
            role_title="Data Scientist",
            seniority_level="Senior",
            required_skills=["Python", "SQL"],
            preferred_skills=["PySpark"],
            domain="FinTech",
            years_experience_min=5,
            behavioral_expectations=["analytical"],
            implicit_signals=["startup experience"],
            responsibilities_summary="Design statistical models."
        )
        emb_text = profile.to_embedding_text()
        self.assertIn("Role: Data Scientist", emb_text)
        self.assertIn("Required skills: Python, SQL", emb_text)

    def test_candidate_profile_embedding_text(self):
        profile = CandidateProfile(
            candidate_id="cand_1",
            name="Alice Smith",
            current_title="Software Engineer",
            explicit_skills=["Python", "SQL"],
            inferred_skills=["Flask"],
            total_years_experience=4.5,
            career_trajectory="Consistent developer",
            domain_exposure=["Finance"],
            behavioral_signals=["team player"],
            platform_signals=["GitHub"],
            education="BS CS"
        )
        emb_text = profile.to_embedding_text()
        self.assertIn("Name: Alice Smith", emb_text)
        self.assertIn("Skills:", emb_text)
        self.assertIn("Flask", emb_text)

    def test_jd_parser_mock(self):
        mock_response = {
            "role_title": "ML Engineer",
            "seniority_level": "Senior",
            "required_skills": ["PyTorch", "Python"],
            "preferred_skills": ["Kubernetes"],
            "domain": "AI",
            "years_experience_min": 5,
            "behavioral_expectations": ["ownership"],
            "implicit_signals": ["papers published"],
            "responsibilities_summary": "Train models."
        }
        client = MockLLMClient(mock_response)
        parser = JDParser(llm_client=client)
        role = parser.parse("Need Senior ML Engineer with PyTorch.")
        self.assertEqual(role.role_title, "ML Engineer")
        self.assertEqual(role.years_experience_min, 5)

    def test_candidate_profiler_mock(self):
        mock_response = {
            "explicit_skills": ["Java"],
            "inferred_skills": ["Spring Boot"],
            "total_years_experience": 3.0,
            "career_trajectory": "Fast growing",
            "domain_exposure": ["SaaS"],
            "behavioral_signals": ["took lead"],
            "platform_signals": ["Active GitHub"],
            "work_history": [
                {
                    "company": "TechCorp",
                    "title": "Developer",
                    "duration_years": 3.0,
                    "description": "Dev work",
                    "domain": "SaaS"
                }
            ]
        }
        client = MockLLMClient(mock_response)
        profiler = CandidateProfiler(llm_client=client, use_llm=True)
        raw_row = {
            "name": "Bob",
            "current_title": "Java Dev",
            "skills": "Java",
            "years_experience": 3,
            "github_url": "github.com/bob"
        }
        profile = profiler.profile_row(raw_row)
        self.assertEqual(profile.name, "Bob")
        self.assertIn("Spring Boot", profile.inferred_skills)
        self.assertEqual(profile.total_years_experience, 3.0)


if __name__ == "__main__":
    unittest.main()
