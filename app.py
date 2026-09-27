from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash
)

import ast
import json
import joblib
import os
import pandas as pd
import sqlite3

from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "change-this-to-a-random-secret-key"
)

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "mental_health_svm_pipeline.pkl"
)

DATABASE = os.path.join(
    BASE_DIR,
    "mental_health_users.db"
)


# =========================================================
# LOAD SAVED MACHINE LEARNING MODEL
# =========================================================

model = joblib.load(MODEL_PATH)


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_db_connection():

    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row

    return connection


# =========================================================
# DATABASE INITIALIZATION + SAFE MIGRATION
# =========================================================

def init_db():

    connection = get_db_connection()
    cursor = connection.cursor()

    # -----------------------------------------------------
    # USERS TABLE
    # -----------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("PRAGMA table_info(users)")
    user_columns = [row["name"] for row in cursor.fetchall()]

    if "password_hash" not in user_columns:
        cursor.execute("""
            ALTER TABLE users
            ADD COLUMN password_hash TEXT
        """)

    if "created_at" not in user_columns:
        cursor.execute("""
            ALTER TABLE users
            ADD COLUMN created_at TIMESTAMP
        """)

    # -----------------------------------------------------
    # Migrate old password column when it exists.
    # The old column may be NOT NULL, so registration below
    # also writes the same secure hash into that column.
    # -----------------------------------------------------

    cursor.execute("PRAGMA table_info(users)")
    user_columns = [row["name"] for row in cursor.fetchall()]

    if "password" in user_columns:

        cursor.execute("""
            SELECT id, password, password_hash
            FROM users
        """)

        for user in cursor.fetchall():

            old_password = user["password"]
            existing_hash = user["password_hash"]

            if old_password and not existing_hash:

                old_password = str(old_password)

                if (
                    old_password.startswith("pbkdf2:")
                    or old_password.startswith("scrypt:")
                ):
                    new_hash = old_password
                else:
                    new_hash = generate_password_hash(old_password)

                cursor.execute("""
                    UPDATE users
                    SET password_hash = ?
                    WHERE id = ?
                """, (new_hash, user["id"]))

    # -----------------------------------------------------
    # ASSESSMENTS TABLE
    # -----------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date_time TEXT,
            risk_level TEXT,
            probability REAL,
            responses TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    cursor.execute("PRAGMA table_info(assessments)")
    assessment_columns = [row["name"] for row in cursor.fetchall()]

    if "user_id" not in assessment_columns:
        cursor.execute("""
            ALTER TABLE assessments
            ADD COLUMN user_id INTEGER
        """)

    if "date_time" not in assessment_columns:
        cursor.execute("""
            ALTER TABLE assessments
            ADD COLUMN date_time TEXT
        """)

    if "risk_level" not in assessment_columns:
        cursor.execute("""
            ALTER TABLE assessments
            ADD COLUMN risk_level TEXT
        """)

    if "probability" not in assessment_columns:
        cursor.execute("""
            ALTER TABLE assessments
            ADD COLUMN probability REAL
        """)

    if "responses" not in assessment_columns:
        cursor.execute("""
            ALTER TABLE assessments
            ADD COLUMN responses TEXT
        """)

    if "created_at" not in assessment_columns:
        cursor.execute("""
            ALTER TABLE assessments
            ADD COLUMN created_at TIMESTAMP
        """)

    connection.commit()
    connection.close()


# =========================================================
# SECURITY HEADERS
# =========================================================

@app.after_request
def add_security_headers(response):

    response.headers["Cache-Control"] = (
        "no-store, no-cache, must-revalidate, max-age=0"
    )
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    return response


# =========================================================
# LOGIN REQUIRED DECORATOR
# =========================================================

def login_required(function):

    @wraps(function)
    def decorated_function(*args, **kwargs):

        if "user_id" not in session:

            flash(
                "Please login to continue.",
                "warning"
            )

            return redirect(url_for("login"))

        return function(*args, **kwargs)

    return decorated_function


# =========================================================
# RECOMMENDATIONS
# =========================================================

def make_recommendation(area, reported, why, tips):
    """Keep the recommendation fields compatible with the templates."""

    return {
        "area": area,
        "what_you_reported": reported,
        "why": why,
        "why_it_matters": why,
        "tips": tips,
        "start": tips[0]
    }


def generate_recommendations(data):

    recommendations = []

    # Sleep and Rest
    if (
        data.get("INSOM") == "Yes"
        or data.get("AVGSLP") in [
            "Below 5h",
            "5-6h",
            "Below 5 hours",
            "5 hours",
            "6 hours"
        ]
    ):
        recommendations.append(make_recommendation(
            "Sleep and Rest",
            "Your responses suggest that your sleep duration or sleep quality may need attention.",
            "Irregular or insufficient sleep can affect energy, concentration, mood and daily functioning.",
            [
                "Try to maintain a regular sleep and wake time.",
                "Reduce screen use shortly before sleeping.",
                "Keep your sleeping environment comfortable and quiet."
            ]
        ))

    # Physical Activity
    if data.get("PHYEX") in ["Never", "Rarely"]:
        recommendations.append(make_recommendation(
            "Physical Activity",
            "You reported doing little or no physical exercise.",
            "Regular physical activity can support overall well-being, energy and stress management.",
            [
                "Start with a short walk each day.",
                "Choose an activity that you can enjoy regularly.",
                "Increase activity gradually rather than making sudden changes."
            ]
        ))

    # Screen and Social Media Use
    if data.get("TSSN") in [
        ">10h",
        "5-7h",
        "5-7 hours a day",
        "More than 10 hours"
    ]:
        recommendations.append(make_recommendation(
            "Screen and Social Media Use",
            "Your reported daily screen or social media use is relatively high.",
            "Long periods of screen use may affect sleep, attention, physical activity and daily routines.",
            [
                "Take short breaks during long screen sessions.",
                "Keep some screen-free periods during the day.",
                "Avoid unnecessary screen use close to bedtime."
            ]
        ))

    # Anxiety and Worry
    if data.get("ANXI") == "Yes":
        recommendations.append(make_recommendation(
            "Anxiety and Worry",
            "You reported experiencing anxiety or frequent worry.",
            "Persistent worry can affect concentration, sleep, energy and everyday activities.",
            [
                "Try slow breathing or short relaxation exercises.",
                "Break stressful tasks into smaller steps.",
                "Talk to someone you trust if worries become difficult to manage."
            ]
        ))

    # Mood and Emotional Well-being
    if data.get("DEPRI") == "Yes":
        recommendations.append(make_recommendation(
            "Mood and Emotional Well-being",
            "You reported experiencing low mood or depressive feelings.",
            "Changes in mood can affect motivation, concentration, sleep and normal daily activities.",
            [
                "Maintain regular daily routines where possible.",
                "Stay connected with supportive people.",
                "Consider speaking with a qualified mental health professional."
            ]
        ))

    # Environment and Surroundings
    if data.get("ENVSAT") == "No":
        recommendations.append(make_recommendation(
            "Environment and Surroundings",
            "You reported being dissatisfied with your current environment.",
            "An uncomfortable or stressful environment can contribute to ongoing stress and affect overall well-being.",
            [
                "Identify specific environmental factors that are causing stress.",
                "Improve your personal space where possible.",
                "Spend some time in places where you feel comfortable and relaxed."
            ]
        ))

    # Financial Stress
    if data.get("FINSTR") == "Yes":
        recommendations.append(make_recommendation(
            "Financial Stress",
            "You reported experiencing financial stress.",
            "Financial concerns can contribute to ongoing worry and may affect sleep, concentration and mood.",
            [
                "Write down your main financial concerns.",
                "Break financial problems into smaller manageable tasks.",
                "Seek appropriate financial guidance when needed."
            ]
        ))

    # Work or Study Pressure
    if data.get("WRKPRE") in ["High", "Very High", "Severe"]:
        recommendations.append(make_recommendation(
            "Work or Study Pressure",
            "You reported experiencing high work or study pressure.",
            "Continuous pressure can contribute to stress, tiredness and difficulty maintaining a healthy routine.",
            [
                "Break large tasks into smaller goals.",
                "Take short breaks during long work or study sessions.",
                "Keep some time aside for rest and personal activities."
            ]
        ))

    # Fallback
    if not recommendations:
        recommendations.append(make_recommendation(
            "Maintaining Healthy Routines",
            "Your responses did not highlight a specific area that needs immediate attention.",
            "Maintaining healthy daily routines can support overall mental and physical well-being.",
            [
                "Maintain a regular sleep schedule.",
                "Stay physically active.",
                "Keep in touch with supportive people.",
                "Make time for activities that help you relax."
            ]
        ))

    return recommendations


# =========================================================
# GET INPUT DATA
# =========================================================

def get_input_data():

    fields = [
        "AGERNG", "GENDER", "EDU", "PROF", "MARSTS", "RESDPL", "LIVWTH",
        "ENVSAT", "POSSAT", "FINSTR", "DEBT", "PHYEX", "SMOKE", "DRINK",
        "ILLNESS", "PREMED", "EATDIS", "AVGSLP", "INSOM", "TSSN", "WRKPRE",
        "ANXI", "DEPRI", "ABUSED", "CHEAT", "THREAT", "SUICIDE", "INFER",
        "CONFLICT", "LOST"
    ]

    return {
        field: request.form.get(field, "")
        for field in fields
    }


# =========================================================
# RISK LEVEL
# =========================================================

def get_risk_level(probability):

    if probability < 0.40:
        return "Low"

    if probability < 0.70:
        return "Moderate"

    return "High"


# =========================================================
# REGISTER
# =========================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username:
            flash("Please enter a username.", "danger")
            return render_template("register.html")

        if not password:
            flash("Please enter a password.", "danger")
            return render_template("register.html")

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("register.html")

        connection = get_db_connection()

        existing_user = connection.execute(
            "SELECT id FROM users WHERE username = ?",
            (username,)
        ).fetchone()

        if existing_user:
            connection.close()
            flash(
                "Username already exists. Please choose another username.",
                "danger"
            )
            return render_template("register.html")

        password_hash = generate_password_hash(password)

        # Support both the current schema and the older schema that
        # contains a NOT NULL `password` column.
        cursor = connection.cursor()
        cursor.execute("PRAGMA table_info(users)")
        columns = [row["name"] for row in cursor.fetchall()]

        if "password" in columns:
            connection.execute(
                """
                INSERT INTO users (username, password, password_hash)
                VALUES (?, ?, ?)
                """,
                (username, password_hash, password_hash)
            )
        else:
            connection.execute(
                """
                INSERT INTO users (username, password_hash)
                VALUES (?, ?)
                """,
                (username, password_hash)
            )

        connection.commit()
        connection.close()

        flash("Account created successfully. Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


# =========================================================
# LOGIN
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        connection = get_db_connection()

        user = connection.execute(
            """
            SELECT id, username, password_hash
            FROM users
            WHERE username = ?
            """,
            (username,)
        ).fetchone()

        connection.close()

        if user is None:
            flash("Username not found.", "danger")
            return render_template("login.html")

        if not user["password_hash"]:
            flash(
                "This account needs to be registered again because its old password could not be migrated.",
                "danger"
            )
            return render_template("login.html")

        if not check_password_hash(user["password_hash"], password):
            flash("Incorrect password.", "danger")
            return render_template("login.html")

        session.clear()
        session["user_id"] = user["id"]
        session["username"] = user["username"]

        return redirect(url_for("index"))

    return render_template("login.html")


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.clear()
    flash("You have been logged out.", "success")

    return redirect(url_for("login"))


# =========================================================
# HOME / ASSESSMENT
# =========================================================

@app.route("/")
@login_required
def index():
    return render_template("index.html")


# =========================================================
# PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
@login_required
def predict():

    input_data = get_input_data()
    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    probability = float(probabilities[1])

    risk_level = get_risk_level(probability)
    recommendations = generate_recommendations(input_data)

    # Save responses as JSON so they can be restored reliably later.
    responses_json = json.dumps(input_data)

    connection = get_db_connection()

    cursor = connection.execute(
        """
        INSERT INTO assessments
        (user_id, risk_level, probability, responses)
        VALUES (?, ?, ?, ?)
        """,
        (
            session["user_id"],
            risk_level,
            probability,
            responses_json
        )
    )

    assessment_id = cursor.lastrowid
    connection.commit()
    connection.close()

    assessment = {
        "id": assessment_id,
        "risk_level": risk_level,
        "probability": probability,
        "prediction": int(prediction),
        "responses": input_data
    }

    # Pass both names so the result template is not dependent on
    # one particular variable naming version.
    return render_template(
        "result.html",
        assessment=assessment,
        risk_level=risk_level,
        recommendations=recommendations
    )


# =========================================================
# DETAILED ANALYSIS
# =========================================================

@app.route("/detailed-analysis", methods=["POST"])
@login_required
def detailed_analysis():

    input_data = get_input_data()
    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    probability = float(probabilities[1])

    risk_level = get_risk_level(probability)
    recommendations = generate_recommendations(input_data)

    assessment = {
        "risk_level": risk_level,
        "probability": probability,
        "prediction": int(prediction),
        "responses": input_data
    }

    return render_template(
        "details.html",
        assessment=assessment,
        risk_level=risk_level,
        recommendations=recommendations,
        history_view=False
    )


# =========================================================
# HISTORY
# =========================================================

@app.route("/history")
@login_required
def history():

    connection = get_db_connection()

    rows = connection.execute(
        """
        SELECT
            id,
            risk_level,
            probability,
            COALESCE(created_at, date_time) AS display_date
        FROM assessments
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (session["user_id"],)
    ).fetchall()

    connection.close()

    assessments = []

    for row in rows:
        assessments.append({
            "id": row["id"],
            "risk_level": row["risk_level"],
            "probability": row["probability"],
            "created_at": row["display_date"] or "Date not available"
        })

    return render_template(
        "history.html",
        assessments=assessments
    )


# =========================================================
# VIEW ONE HISTORY ASSESSMENT
# =========================================================

@app.route("/history/<int:assessment_id>")
@login_required
def history_detail(assessment_id):

    connection = get_db_connection()

    # Ownership is checked on the server. A user cannot access
    # another user's assessment by changing the URL number.
    assessment = connection.execute(
        """
        SELECT
            id,
            user_id,
            risk_level,
            probability,
            responses,
            COALESCE(created_at, date_time) AS display_date
        FROM assessments
        WHERE id = ?
        AND user_id = ?
        """,
        (
            assessment_id,
            session["user_id"]
        )
    ).fetchone()

    connection.close()

    if assessment is None:
        return "Assessment not found.", 404

    responses = {}
    saved_responses = assessment["responses"]

    if saved_responses:
        try:
            responses = json.loads(saved_responses)
        except Exception:
            try:
                responses = ast.literal_eval(saved_responses)
            except Exception:
                responses = {}

    recommendations = generate_recommendations(responses)

    assessment_data = {
        "id": assessment["id"],
        "risk_level": assessment["risk_level"],
        "probability": assessment["probability"],
        "responses": responses,
        "created_at": assessment["display_date"] or "Date not available"
    }

    return render_template(
        "details.html",
        assessment=assessment_data,
        risk_level=assessment_data["risk_level"],
        recommendations=recommendations,
        history_view=True
    )


# =========================================================
# INITIALIZE DATABASE
# =========================================================

init_db()


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(debug=True)
