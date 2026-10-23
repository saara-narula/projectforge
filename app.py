import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from recommender import ProjectRecommender

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key-12345")

recommender = ProjectRecommender()

@app.route("/")
def index():
    metadata = recommender.get_metadata_options()
    presets = [
        {
            "name": "AI & Deep Learning Aspirant",
            "background": "Computer Science",
            "domains": ["Artificial Intelligence", "Computer Vision"],
            "tech": ["Python", "PyTorch", "OpenCV"],
            "difficulty": "Intermediate",
            "scope": "Capstone Project",
            "interests": "neural networks medical imaging computer vision deep learning"
        },
        {
            "name": "Full-Stack Web Craftsman",
            "background": "Information Technology",
            "domains": ["Web Development", "Cloud & Distributed Systems"],
            "tech": ["Python", "Flask", "HTML", "CSS", "JavaScript"],
            "difficulty": "Intermediate",
            "scope": "Portfolio Booster",
            "interests": "websockets real-time microservices responsive web apps"
        },
        {
            "name": "Cybersecurity & Forensic Analyst",
            "background": "Cybersecurity & Information Security",
            "domains": ["Cybersecurity", "Network Security"],
            "tech": ["Python", "Socket Programming", "Cryptography"],
            "difficulty": "Intermediate",
            "scope": "Portfolio Booster",
            "interests": "penetration testing steganography vulnerability auditing network analysis"
        },
        {
            "name": "Hardware, IoT & Robotics Engineer",
            "background": "Electrical & Electronics",
            "domains": ["Internet of Things (IoT)", "Robotics & Automation"],
            "tech": ["Python", "MQTT", "C++", "NumPy"],
            "difficulty": "Intermediate",
            "scope": "Capstone Project",
            "interests": "smart sensors microcontrollers automation teleoperation kinematics"
        },
        {
            "name": "FinTech & Data Science Analyst",
            "background": "Data Science & AI",
            "domains": ["FinTech", "Data Science & Analytics"],
            "tech": ["Python", "Pandas", "NumPy", "Scikit-Learn"],
            "difficulty": "Intermediate",
            "scope": "Portfolio Booster",
            "interests": "algorithmic trading quantitative finance fraud detection analytics"
        }
    ]
    return render_template("index.html", metadata=metadata, presets=presets)

@app.route("/recommend", methods=["GET", "POST"])
def recommend_view():
    if request.method == "POST":
        academic_background = request.form.get("academic_background", "").strip()
        domains = request.form.getlist("domains")
        tech_stack_raw = request.form.getlist("tech_stack")
        tech_custom = request.form.get("tech_custom", "").strip()
        tech_list = list(tech_stack_raw)
        if tech_custom:
            tech_list.extend([t.strip() for t in tech_custom.split(",") if t.strip()])

        difficulty = request.form.get("difficulty", "Any")
        scope = request.form.get("scope", "Any")
        interests = request.form.get("interests", "").strip()
        algorithm_mode = request.form.get("algorithm_mode", "hybrid")
    else:
        academic_background = request.args.get("background", "").strip()
        domains = request.args.getlist("domain")
        tech_list = request.args.getlist("tech")
        difficulty = request.args.get("difficulty", "Any")
        scope = request.args.get("scope", "Any")
        interests = request.args.get("interests", "").strip()
        algorithm_mode = request.args.get("mode", "hybrid")

    recommendations = recommender.recommend(
        academic_background=academic_background,
        domains=domains,
        tech_stack=tech_list,
        difficulty=difficulty,
        scope=scope,
        interests=interests,
        algorithm_mode=algorithm_mode,
        top_n=15
    )

    query_params = {
        "academic_background": academic_background or "All Majors",
        "domains": domains or ["All Domains"],
        "tech_stack": tech_list or ["General / Any"],
        "difficulty": difficulty,
        "scope": scope,
        "interests": interests,
        "algorithm_mode": algorithm_mode
    }

    return render_template("results.html", 
                           recommendations=recommendations, 
                           query=query_params,
                           metadata=recommender.get_metadata_options())

@app.route("/explore")
def explore():
    all_projects = recommender.get_all_projects()
    metadata = recommender.get_metadata_options()
    
    domain_filter = request.args.get("domain", "")
    difficulty_filter = request.args.get("difficulty", "")
    scope_filter = request.args.get("scope", "")
    search_query = request.args.get("q", "").lower().strip()

    filtered = []
    for p in all_projects:
        if domain_filter and domain_filter not in p.get("domains", []):
            continue
        if difficulty_filter and p.get("difficulty") != difficulty_filter:
            continue
        if scope_filter and p.get("scope") != scope_filter:
            continue
        if search_query:
            searchable = (p.get("title", "") + " " + p.get("tagline", "") + " " + 
                          p.get("description", "") + " " + " ".join(p.get("tech_stack", [])) + 
                          " " + " ".join(p.get("tags", []))).lower()
            if search_query not in searchable:
                continue
        
        # Attach community stats
        p_copy = dict(p)
        stats = recommender.get_project_community_stats(p["id"])
        p_copy["avg_rating"] = stats["avg_rating"]
        p_copy["review_count"] = stats["review_count"]
        filtered.append(p_copy)

    return render_template("explore.html", 
                           projects=filtered, 
                           metadata=metadata,
                           active_domain=domain_filter,
                           active_difficulty=difficulty_filter,
                           active_scope=scope_filter,
                           search_query=search_query)

@app.route("/project/<project_id>")
def project_detail(project_id):
    project = recommender.get_project_by_id(project_id)
    if not project:
        return render_template("404.html", message=f"Project '{project_id}' not found"), 404
    
    # 1. Content-based related projects
    related = recommender.recommend(
        academic_background=project.get("academic_backgrounds", [""])[0],
        domains=project.get("domains", []),
        tech_stack=project.get("tech_stack", []),
        difficulty=project.get("difficulty", "Any"),
        algorithm_mode="content",
        top_n=4
    )
    related = [r for r in related if r["project"]["id"] != project_id][:3]

    # 2. Collaborative Filtering: "Students who liked this also worked on..."
    collab_similar = recommender.get_collaborative_similar_items(project_id, top_k=3)

    # 3. Community Rating Stats
    stats = recommender.get_project_community_stats(project_id)

    return render_template("project_detail.html", 
                           project=project, 
                           related=related,
                           collab_similar=collab_similar,
                           stats=stats)

@app.route("/about")
def about():
    return render_template("about.html")

# ==========================================
# REST API Endpoints
# ==========================================
@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    data = request.get_json() or {}
    results = recommender.recommend(
        academic_background=data.get("academic_background", ""),
        domains=data.get("domains", []),
        tech_stack=data.get("tech_stack", []),
        difficulty=data.get("difficulty", "Any"),
        scope=data.get("scope", "Any"),
        interests=data.get("interests", ""),
        algorithm_mode=data.get("algorithm_mode", "hybrid"),
        top_n=data.get("top_n", 10)
    )
    return jsonify({"success": True, "count": len(results), "recommendations": results})

@app.route("/api/rate", methods=["POST"])
def api_rate():
    """Allows users to submit a live star rating for collaborative filtering."""
    data = request.get_json() or {}
    project_id = data.get("project_id")
    rating = data.get("rating")
    major = data.get("major", "General STEM")

    if not project_id or rating is None:
        return jsonify({"success": False, "error": "Missing project_id or rating"}), 400

    updated_stats = recommender.record_user_rating(project_id, rating, major)
    return jsonify({
        "success": True, 
        "project_id": project_id, 
        "updated_stats": updated_stats,
        "message": "Rating recorded and collaborative filtering matrices updated!"
    })

@app.route("/api/projects")
def api_projects():
    return jsonify({"success": True, "projects": recommender.get_all_projects()})

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
