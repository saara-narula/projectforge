import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

TECH_ALIASES = {
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "reactjs": "react",
    "vuejs": "vue",
    "k8s": "kubernetes",
    "postgres": "postgresql",
    "golang": "go",
    "tf": "tensorflow",
    "sklearn": "scikit-learn",
    "mongo": "mongodb",
    "cplusplus": "c++",
    "cpp": "c++",
    "dockerized": "docker",
    "kube": "kubernetes",
    "kafka": "apache kafka",
    "spark": "apache spark"
}

class ProjectRecommender:
    """
    True Hybrid Recommendation Engine combining:
    1. Content-Based Multi-Attribute Model (Academic discipline, tech stack, domains, difficulty)
    2. Content-Based TF-IDF Cosine Similarity (Semantic intent & research keywords)
    3. Item-Item Collaborative Filtering (Co-occurrence & cosine similarity on student rating matrix)
    4. User-Based Collaborative Filtering (Peer neighborhood preferences & major cohort trends)
    """

    def __init__(self, data_path: str = None, interactions_path: str = None):
        base_dir = os.path.dirname(__file__)
        if data_path is None:
            data_path = os.path.join(base_dir, "data", "projects.json")
        if interactions_path is None:
            interactions_path = os.path.join(base_dir, "data", "user_interactions.json")

        self.data_path = data_path
        self.interactions_path = interactions_path

        self.projects = self._load_projects()
        self.interactions = self._load_interactions()
        self._build_vocab_and_idf()
        self._build_collaborative_matrices()

    def _load_projects(self) -> List[Dict[str, Any]]:
        with open(self.data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_interactions(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.interactions_path):
            with open(self.interactions_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def save_interactions(self):
        with open(self.interactions_path, "w", encoding="utf-8") as f:
            json.dump(self.interactions, f, indent=2)
        # Rebuild matrices when new ratings arrive
        self._build_collaborative_matrices()

    def record_user_rating(self, project_id: str, rating: float, major: str = "General STEM") -> Dict[str, Any]:
        """Logs a live student rating into the interaction dataset."""
        new_record = {
            "user_id": f"web_user_{len(self.interactions) + 1}",
            "major": major or "General STEM",
            "project_id": project_id,
            "rating": max(1.0, min(5.0, float(rating))),
            "timestamp": "2026-10-16"
        }
        self.interactions.append(new_record)
        self.save_interactions()
        return self.get_project_community_stats(project_id)

    # ==========================================
    # Collaborative Filtering Matrices & Logic
    # ==========================================
    def _build_collaborative_matrices(self):
        """
        Builds:
        1. user_ratings: {user_id: {project_id: rating}}
        2. item_ratings: {project_id: {user_id: rating}}
        3. item_similarity: {p1: {p2: cosine_similarity}}
        4. major_preferences: {major: Counter(project_ids weighted by rating)}
        """
        self.user_ratings = defaultdict(dict)
        self.item_ratings = defaultdict(dict)
        self.major_preferences = defaultdict(lambda: defaultdict(list))
        self.project_stats = {}

        for record in self.interactions:
            uid = record["user_id"]
            pid = record["project_id"]
            r = float(record["rating"])
            m = record.get("major", "General STEM").strip().lower()

            self.user_ratings[uid][pid] = r
            self.item_ratings[pid][uid] = r
            self.major_preferences[m][pid].append(r)

        # Precompute community rating averages & counts for each project
        for p in self.projects:
            pid = p["id"]
            ratings = list(self.item_ratings.get(pid, {}).values())
            if ratings:
                avg_rating = sum(ratings) / len(ratings)
                review_count = len(ratings)
            else:
                avg_rating = 4.5  # default baseline for unrated items
                review_count = 0

            self.project_stats[pid] = {
                "avg_rating": round(avg_rating, 1),
                "review_count": review_count
            }

        # Compute Item-Item Cosine Similarity Matrix
        self.item_similarity = defaultdict(dict)
        project_ids = [p["id"] for p in self.projects]

        for i, p1 in enumerate(project_ids):
            for j, p2 in enumerate(project_ids):
                if i == j:
                    self.item_similarity[p1][p2] = 1.0
                    continue

                r1 = self.item_ratings.get(p1, {})
                r2 = self.item_ratings.get(p2, {})
                common_users = set(r1.keys()) & set(r2.keys())

                if not common_users:
                    self.item_similarity[p1][p2] = 0.0
                else:
                    dot = sum(r1[u] * r2[u] for u in common_users)
                    norm1 = math.sqrt(sum(r1[u] ** 2 for u in common_users))
                    norm2 = math.sqrt(sum(r2[u] ** 2 for u in common_users))
                    if norm1 > 0 and norm2 > 0:
                        self.item_similarity[p1][p2] = dot / (norm1 * norm2)
                    else:
                        self.item_similarity[p1][p2] = 0.0

    def get_project_community_stats(self, project_id: str) -> Dict[str, Any]:
        return self.project_stats.get(project_id, {"avg_rating": 4.5, "review_count": 0})

    def get_collaborative_similar_items(self, project_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Item-Item Collaborative Filtering:
        Returns projects that students also loved who liked this project.
        """
        sims = self.item_similarity.get(project_id, {})
        # Sort by similarity descending
        ranked = sorted(
            [(pid, sim) for pid, sim in sims.items() if pid != project_id and sim > 0],
            key=lambda x: x[1],
            reverse=True
        )

        results = []
        for pid, sim in ranked[:top_k]:
            proj = self.get_project_by_id(pid)
            if proj:
                stats = self.get_project_community_stats(pid)
                results.append({
                    "project": proj,
                    "similarity": round(sim, 2),
                    "avg_rating": stats["avg_rating"],
                    "review_count": stats["review_count"]
                })
        return results

    def _compute_collaborative_score(self, project_id: str, academic_background: str) -> Tuple[float, List[str]]:
        """
        Calculates User-Based and Item-Item Collaborative Filtering score (0 to 100)
        based on peer student interaction vectors, cohort ratings, and co-rating affinity.
        """
        acad = academic_background.strip().lower()
        stats = self.get_project_community_stats(project_id)
        avg_r = stats["avg_rating"]
        count = stats["review_count"]

        reasons = []

        # 1. Base peer community score (normalized 1.0 to 5.0 -> 0 to 100)
        base_score = (avg_r / 5.0) * 100.0

        # 2. Major Cohort Affinity (User-based neighborhood)
        cohort_bonus = 0.0
        if acad and acad in self.major_preferences:
            cohort_pids = self.major_preferences[acad]
            if project_id in cohort_pids:
                cohort_ratings = cohort_pids[project_id]
                cohort_avg = sum(cohort_ratings) / len(cohort_ratings)
                cohort_count = len(cohort_ratings)
                if cohort_avg >= 4.5:
                    cohort_bonus = 18.0
                    reasons.append(f"Rated {cohort_avg:.1f}★ by {cohort_count} peers in your major ({academic_background})")
                else:
                    cohort_bonus = 10.0
                    reasons.append(f"Endorsed by {cohort_count} students in {academic_background}")

        # 3. Item-Item Co-Rating Affinity Boost
        item_sims = self.item_similarity.get(project_id, {})
        high_sim_count = sum(1 for p_other, s in item_sims.items() if p_other != project_id and s >= 0.70)
        if high_sim_count > 0:
            reasons.append(f"Strong co-completion synergy with {high_sim_count} top student portfolio projects")

        if count >= 3:
            reasons.append(f"Community favorite: {avg_r}★ across {count} verified student reviews")

        # Collaborative Score combining general peer rating and cohort preference
        final_cf_score = min(99.0, max(30.0, base_score * 0.80 + cohort_bonus))
        return final_cf_score, reasons

    # ==========================================
    # Text Processing & Metadata Options
    # ==========================================
    def get_all_projects(self) -> List[Dict[str, Any]]:
        return self.projects

    def get_project_by_id(self, project_id: str) -> Dict[str, Any]:
        for p in self.projects:
            if p["id"] == project_id:
                return p
        return None

    def get_metadata_options(self) -> Dict[str, Any]:
        academic_backgrounds = set()
        domains = set()
        tech_stack = set()
        difficulties = ["Beginner", "Intermediate", "Advanced"]
        scopes = ["Mini Project", "Portfolio Booster", "Capstone Project", "Hackathon"]

        for p in self.projects:
            academic_backgrounds.update(p.get("academic_backgrounds", []))
            domains.update(p.get("domains", []))
            tech_stack.update(p.get("tech_stack", []))

        categorized_tech = {
            "Languages": [
                "Python", "JavaScript", "TypeScript", "C++", "Go", "Rust", "Dart", "Solidity"
            ],
            "Web & Backend": [
                "Flask", "FastAPI", "React", "Next.js", "WebSockets", "Tailwind CSS", "GraphQL", "Celery", "HTML", "CSS"
            ],
            "AI & Data Science": [
                "PyTorch", "TensorFlow", "Scikit-Learn", "Pandas", "NumPy", "OpenCV", "MediaPipe", "LangChain", "Ollama", "ChromaDB", "NLTK"
            ],
            "Databases & Streaming": [
                "PostgreSQL", "MongoDB", "Redis", "SQLite", "Apache Kafka", "Apache Spark", "Firebase"
            ],
            "Cloud, DevOps & Systems": [
                "Docker", "Kubernetes", "Prometheus", "Terraform", "Helm", "Linux", "gRPC"
            ],
            "Hardware & Security": [
                "MQTT", "ESP32", "Raspberry Pi", "ROS 2", "Scapy", "Wireshark", "Socket Programming"
            ]
        }

        for cat_list in categorized_tech.values():
            tech_stack.update(cat_list)

        return {
            "academic_backgrounds": sorted(list(academic_backgrounds)),
            "domains": sorted(list(domains)),
            "tech_stack": sorted(list(tech_stack), key=lambda s: s.lower()),
            "categorized_tech": categorized_tech,
            "difficulties": difficulties,
            "scopes": scopes
        }

    @staticmethod
    def _normalize_tech(tech_name: str) -> str:
        clean = tech_name.strip().lower()
        return TECH_ALIASES.get(clean, clean)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        STOP_WORDS = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
            "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
            "by", "can", "did", "do", "does", "doing", "don", "down", "during", "each", "few", "for",
            "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
            "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
            "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
            "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she",
            "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
            "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
            "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
            "why", "with", "you", "your", "yours", "yourself", "yourselves"
        }
        words = re.findall(r"\b[a-zA-Z0-9_\-#\+\.]{2,}\b", text.lower())
        return [w for w in words if w not in STOP_WORDS]

    def _get_project_corpus_text(self, p: Dict[str, Any]) -> str:
        parts = [
            p.get("title", ""),
            p.get("tagline", ""),
            p.get("description", ""),
            " ".join(p.get("domains", [])),
            " ".join(p.get("academic_backgrounds", [])),
            " ".join(p.get("tech_stack", [])),
            " ".join(p.get("tags", []))
        ]
        return " ".join(parts)

    def _build_vocab_and_idf(self):
        self.doc_count = len(self.projects)
        self.doc_freqs = Counter()
        self.project_token_counters = []

        for p in self.projects:
            text = self._get_project_corpus_text(p)
            tokens = self._tokenize(text)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1
            self.project_token_counters.append(Counter(tokens))

        self.idf = {}
        for token, freq in self.doc_freqs.items():
            self.idf[token] = math.log((self.doc_count + 1) / (freq + 1)) + 1.0

    def _compute_tfidf_vector(self, counter: Counter) -> Dict[str, float]:
        vec = {}
        total = sum(counter.values()) or 1
        for word, count in counter.items():
            tf = count / total
            idf = self.idf.get(word, 1.0)
            vec[word] = tf * idf
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            for k in vec:
                vec[k] /= norm
        return vec

    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        if not vec1 or not vec2:
            return 0.0
        common_keys = set(vec1.keys()) & set(vec2.keys())
        return sum(vec1[k] * vec2[k] for k in common_keys)

    # ==========================================
    # Hybrid Recommendation Engine
    # ==========================================
    def recommend(
        self,
        academic_background: str = "",
        domains: List[str] = None,
        tech_stack: List[str] = None,
        difficulty: str = "Any",
        scope: str = "Any",
        interests: str = "",
        algorithm_mode: str = "hybrid",
        top_n: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Hybrid Recommendation Engine (Collaborative Filtering + Content-Based Filtering):
        - algorithm_mode="hybrid" (default): 50% Content-Based + 50% Collaborative Filtering
        - algorithm_mode="collaborative": 85% Collaborative Filtering + 15% Content-compatibility
        - algorithm_mode="content": 100% Content-Based
        """
        domains = [d.strip().lower() for d in (domains or []) if d.strip()]
        user_skills = [self._normalize_tech(t) for t in (tech_stack or []) if t.strip()]
        user_skills_set = set(user_skills)
        user_acad = academic_background.strip().lower()
        diff_pref = difficulty.strip().lower()
        scope_pref = scope.strip().lower()
        mode = algorithm_mode.strip().lower()

        user_interest_tokens = self._tokenize(interests)
        user_tfidf = None
        if user_interest_tokens:
            user_tfidf = self._compute_tfidf_vector(Counter(user_interest_tokens))

        scored_projects = []

        for idx, project in enumerate(self.projects):
            pid = project["id"]
            proj_acad = [a.lower() for a in project.get("academic_backgrounds", [])]
            proj_domains = [d.lower() for d in project.get("domains", [])]
            proj_stack_orig = project.get("tech_stack", [])
            proj_diff = project.get("difficulty", "").lower()
            proj_scope = project.get("scope", "").lower()

            match_reasons = []

            # ------------------------------------
            # 1. Content-Based Scoring
            # ------------------------------------
            # Academic fit
            acad_score = 0.0
            if not user_acad or user_acad == "any" or user_acad == "general stem / other":
                acad_score = 0.85
            else:
                for a in proj_acad:
                    if user_acad in a or a in user_acad:
                        acad_score = 1.0
                        match_reasons.append(f"Highly relevant for {project.get('academic_backgrounds', ['your major'])[0]} students")
                        break
                if acad_score == 0.0:
                    acad_score = 0.45

            # Domain alignment
            domain_score = 0.0
            if not domains:
                domain_score = 0.75
            else:
                matched_domains = []
                for ud in domains:
                    for pd in proj_domains:
                        if ud in pd or pd in ud:
                            matched_domains.append(pd)
                            break
                if matched_domains:
                    ratio = len(matched_domains) / len(domains)
                    domain_score = min(1.0, 0.5 + 0.5 * ratio)
                    match_reasons.append(f"Matches your target domain: {matched_domains[0].title()}")
                else:
                    domain_score = 0.25

            # Tech stack overlap
            matched_skills = []
            learning_skills = []
            for skill in proj_stack_orig:
                norm_proj_skill = self._normalize_tech(skill)
                is_match = False
                for us in user_skills_set:
                    if us in norm_proj_skill or norm_proj_skill in us:
                        is_match = True
                        break
                if is_match:
                    matched_skills.append(skill)
                else:
                    learning_skills.append(skill)

            if not user_skills_set:
                tech_score = 0.75
            else:
                if matched_skills:
                    overlap_ratio = len(matched_skills) / max(len(proj_stack_orig), 1)
                    tech_score = min(1.0, 0.4 + 0.6 * overlap_ratio)
                    match_reasons.append(f"Builds on your tech stack: {', '.join(matched_skills[:3])}")
                else:
                    tech_score = 0.25

            # Difficulty & Scope fit
            diff_score = 1.0
            if diff_pref and diff_pref != "any":
                diff_hierarchy = {"beginner": 1, "intermediate": 2, "advanced": 3}
                user_val = diff_hierarchy.get(diff_pref, 2)
                proj_val = diff_hierarchy.get(proj_diff, 2)
                diff_delta = abs(user_val - proj_val)
                diff_score = 1.0 if diff_delta == 0 else (0.7 if diff_delta == 1 else 0.35)

            scope_score = 1.0
            if scope_pref and scope_pref != "any":
                if scope_pref in proj_scope or proj_scope in scope_pref:
                    scope_score = 1.0
                    match_reasons.append(f"Fits your target scope: {project.get('scope')}")
                else:
                    scope_score = 0.6

            diff_scope_score = (diff_score * 0.6) + (scope_score * 0.4)

            # TF-IDF semantic match
            tfidf_score = 0.5
            if user_tfidf:
                proj_tfidf = self._compute_tfidf_vector(self.project_token_counters[idx])
                sim = self._cosine_similarity(user_tfidf, proj_tfidf)
                tfidf_score = min(1.0, sim * 2.5)
                if sim > 0.15:
                    match_reasons.append("Strong semantic alignment with your stated research interests")

            content_raw_score = (
                (domain_score * 0.25) +
                (tech_score * 0.25) +
                (acad_score * 0.18) +
                (diff_scope_score * 0.17) +
                (tfidf_score * 0.15)
            ) * 100.0

            # ------------------------------------
            # 2. Collaborative Filtering Scoring
            # ------------------------------------
            collab_score, collab_reasons = self._compute_collaborative_score(pid, academic_background)

            # ------------------------------------
            # 3. Hybrid Scoring (Collaborative + Content-Based Ensemble)
            # ------------------------------------
            if mode == "content":
                final_score = content_raw_score
                if collab_reasons:
                    match_reasons.extend(collab_reasons)
            elif mode == "collaborative":
                final_score = (collab_score * 0.85) + (content_raw_score * 0.15)
                if collab_reasons:
                    match_reasons = collab_reasons + match_reasons
            else:  # hybrid (default: collaborative as well as content based)
                final_score = (content_raw_score * 0.50) + (collab_score * 0.50)
                # Seamlessly integrate both collaborative peer endorsements and content prerequisites
                if collab_reasons:
                    match_reasons = [collab_reasons[0]] + match_reasons + collab_reasons[1:]

            final_percentage = max(25, min(99, int(round(final_score))))
            stats = self.get_project_community_stats(pid)

            if not match_reasons:
                match_reasons.append(f"Recommended {project.get('difficulty')} {project.get('scope')}")
                match_reasons.append(f"Features technologies: {', '.join(project.get('tech_stack', [])[:3])}")

            scored_projects.append({
                "project": project,
                "score": final_percentage,
                "content_score": int(content_raw_score),
                "collaborative_score": int(collab_score),
                "avg_rating": stats["avg_rating"],
                "review_count": stats["review_count"],
                "match_reasons": match_reasons[:3],
                "matched_skills": matched_skills,
                "learning_skills": learning_skills,
                "breakdown": {
                    "domain_score": int(domain_score * 100),
                    "tech_score": int(tech_score * 100),
                    "academic_score": int(acad_score * 100),
                    "difficulty_score": int(diff_scope_score * 100),
                    "semantic_score": int(tfidf_score * 100),
                    "collaborative_score": int(collab_score)
                }
            })

        scored_projects.sort(key=lambda x: x["score"], reverse=True)
        return scored_projects[:top_n]
