import pandas as pd
import numpy as np
import json
import sys
import os

# Optional imports for training mode
try:
    from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, classification_report
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from imblearn.over_sampling import SMOTE
    from collections import Counter
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Paths relative to this script
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback when __file__ is not defined
    SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv else os.getcwd()

EXCEL_DIR = os.path.join(SCRIPT_DIR, 'excel')
MODEL_OUTPUT = os.path.join(SCRIPT_DIR, 'strandalign_model.joblib')
SCALER_OUTPUT = os.path.join(SCRIPT_DIR, 'strandalign_scaler.joblib')
FEATURES_OUTPUT = os.path.join(SCRIPT_DIR, 'strandalign_features.json')
FEATURE_IMPORTANCE_OUTPUT = os.path.join(SCRIPT_DIR, 'feature_importances.csv')
RANDOM_STATE = 42
N_SPLITS = 5


def generate_strand_breakdown(group_label):
    if group_label == "Academic":
        weights = {
            "STEM": np.random.uniform(20, 35),
            "ABM": np.random.uniform(15, 30),
            "HUMSS": np.random.uniform(15, 30),
            "GAS": np.random.uniform(15, 30),
        }
    elif group_label == "TechPro":
        weights = {
            "ICT": np.random.uniform(20, 35),
            "HE": np.random.uniform(15, 30),
            "IA": np.random.uniform(15, 30),
            "Caregiving": np.random.uniform(15, 30),
        }
    else:
        weights = {
            "STEM": np.random.uniform(20, 25),
            "ABM": np.random.uniform(20, 25),
            "HUMSS": np.random.uniform(20, 25),
            "GAS": np.random.uniform(20, 25),
        }
    total = sum(weights.values())
    return {k: round((v / total) * 100, 2) for k, v in weights.items()}


def simple_rule_prediction(row, acad_col, tvl_col):
    a = float(row.get(acad_col, 0) or 0)
    t = float(row.get(tvl_col, 0) or 0)
    
    diff = a - t
    # Very tight margin for "Undecided" - only when scores are extremely close (within 0.2)
    neutral_margin = 0.2
    
    # Check if scores are truly neutral (very close)
    if abs(diff) <= neutral_margin:
        # Only use Undecided if both scores are meaningful (>= 2.5) or both are low
        if (a >= 2.5 and t >= 2.5) or (a < 2.5 and t < 2.5):
            label = "Undecided"
            # Low confidence for truly neutral cases
            conf = max(0.50, min(0.60, 0.5 + abs(diff) * 0.5))
        else:
            # Even if close, if one is clearly above threshold, pick it
            if a > t:
                label = "Academic"
                conf = max(0.55, min(0.70, 0.5 + diff / 5))
            else:
                label = "TechPro"
                conf = max(0.55, min(0.70, 0.5 + abs(diff) / 5))
    else:
        # Clear preference - pick the higher score
        if diff > neutral_margin:
            label = "Academic"
            if a >= 3 and t < 3:
                conf = max(0.75, min(0.98, 0.7 + diff / 4))  # High confidence
            elif a >= 3 and t >= 3:
                conf = max(0.65, min(0.85, 0.6 + diff / 5))  # Medium confidence
            else:
                conf = max(0.55, min(0.75, 0.5 + diff / 6))  # Lower confidence
        else:
            label = "TechPro"
            if t >= 3 and a < 3:
                conf = max(0.75, min(0.98, 0.7 + abs(diff) / 4))  # High confidence
            elif t >= 3 and a >= 3:
                conf = max(0.65, min(0.85, 0.6 + abs(diff) / 5))  # Medium confidence
            else:
                conf = max(0.55, min(0.75, 0.5 + abs(diff) / 6))  # Lower confidence
    
    return label, round(conf, 2)


def find_label_columns(df):
    """Find Academic and TechPro track label columns with flexible matching."""
    cols = list(df.columns)
    low = [c.lower() for c in cols]

    acad_candidates = [cols[i] for i, c in enumerate(low)
                       if "academic track" in c
                       or ("academic" in c and "track" in c)]
    tech_candidates = [cols[i] for i, c in enumerate(low)
                       if "technical-professional" in c
                       or "technical professional" in c
                       or ("technical" in c and "track" in c)
                       or "technical-professional track" in c]

    # Broader search if not found
    if not acad_candidates:
        acad_candidates = [cols[i] for i, c in enumerate(low) 
                          if "academic" in c]
    if not tech_candidates:
        tech_candidates = [cols[i] for i, c in enumerate(low) 
                          if "technical" in c or "tvl" in c]

    return acad_candidates, tech_candidates


def remove_tagalog_columns(df, preserve_cols=None):
    """Remove Tagalog duplicate columns while preserving specified columns."""
    if preserve_cols is None:
        preserve_cols = set()
    
    tagalog_patterns = [
        "interesado ako", "gusto ko", "mas gusto kong", "magaling akong",
        "kumpiyansa akong", "malikhain ako", "nababantayan ko", "kaya kong",
        "mabilis akong", "mas natututo ako", "pakiramdam ko", "karaniwan akong",
        "interesado akong", "gusto kong"
    ]
    
    pattern_regex = "|".join(tagalog_patterns)
    match_mask = df.columns.str.contains(pattern_regex, case=False, na=False)
    preserve_mask = df.columns.isin(preserve_cols)
    drop_mask = match_mask & (~preserve_mask)
    
    return df.loc[:, ~drop_mask]


def prepare_features(df, drop_cols=None, is_training=True, scaler=None, training_medians=None):
    """
    Prepare features with binarized preferences and normalization.
    Returns: (X, grade_cols, preference_cols, scaler, feature_names)
    """
    if drop_cols is None:
        drop_cols = []
    
    # Feature selection
    grade_cols = [c for c in df.columns if any(g in c for g in ["Grade 7", "Grade 8", "Grade 9", "Grade 10"])]
    subject_interest_cols = [c for c in df.columns if "I am interested in" in c]
    ability_cols = [c for c in df.columns if ("I am" in c or "I can" in c or "I pay attention" in c)]
    enjoyment_cols = [c for c in df.columns if ("I enjoy" in c or "I learn best" in c)]
    preference_cols = [c for c in df.columns if ("I prefer" in c or "I want" in c)]
    
    feature_cols = grade_cols + subject_interest_cols + preference_cols + ability_cols + enjoyment_cols
    if not feature_cols:
        feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols].copy()
    
    # Numeric conversion
    X = X.apply(pd.to_numeric, errors="coerce")
    
    # Fill missing values
    if is_training:
        training_medians = X.median(numeric_only=True)
        X = X.fillna(training_medians)
    else:
        if training_medians is not None:
            for col in X.columns:
                if col in training_medians:
                    X[col] = X[col].fillna(training_medians[col])
                else:
                    X[col] = X[col].fillna(0)
        else:
            X = X.fillna(0)
    
    # Binarize preference columns: >=4 -> 1, else -> 0
    for col in preference_cols:
        if col in X.columns:
            X[col] = (X[col] >= 4).astype(int)
    
    # Normalize grade columns
    if grade_cols:
        if is_training:
            scaler = StandardScaler()
            X[grade_cols] = scaler.fit_transform(X[grade_cols])
        else:
            if scaler is not None:
                X[grade_cols] = scaler.transform(X[grade_cols])
    
    return X, grade_cols, preference_cols, scaler, list(X.columns), training_medians


def run_prediction_mode(input_csv_path: str) -> int:
    if not ML_AVAILABLE:
        print(json.dumps({'error': 'ML libraries not available'}), file=sys.stderr)
        return 1
    
    try:
        df_in = pd.read_csv(input_csv_path, encoding='utf-8')
    except Exception:
        try:
            df_in = pd.read_csv(input_csv_path, encoding='latin1')
        except Exception as e:
            print(json.dumps({'error': f'Failed to read CSV: {str(e)}'}), file=sys.stderr)
            return 1
    
    print(f"DEBUG: Loaded CSV with {len(df_in)} rows and columns: {list(df_in.columns)}", file=sys.stderr)

    # Detect label columns for fallback
    acad_candidates, tvl_candidates = find_label_columns(df_in)
    acad_col = acad_candidates[0] if acad_candidates else None
    tvl_col = tvl_candidates[0] if tvl_candidates else None

    # Helpers to fetch common fields with flexible header names
    def pick_case_insensitive(row_like, candidates):
        for c in row_like.index if hasattr(row_like, 'index') else row_like.keys():
            col_name = str(c).strip().lower()
            for cand in candidates:
                if col_name == cand.lower() or col_name.replace('_', ' ').replace('-', ' ') == cand.lower().replace('_', ' ').replace('-', ' '):
                    return row_like.get(c)
        return None

    results = []
    
    # Try to use model if available
    model_available = os.path.exists(MODEL_OUTPUT)
    scaler_available = os.path.exists(SCALER_OUTPUT)
    features_available = os.path.exists(FEATURES_OUTPUT)
    
    clf = None
    scaler = None
    training_features = None
    training_medians = None
    
    if model_available and scaler_available and features_available:
        try:
            clf = joblib.load(MODEL_OUTPUT)
            scaler = joblib.load(SCALER_OUTPUT)
            with open(FEATURES_OUTPUT, 'r') as f:
                feature_data = json.load(f)
                training_features = feature_data.get('feature_names', [])
                training_medians = pd.Series(feature_data.get('medians', {}))
            print(f"DEBUG: Loaded model with {len(training_features)} features", file=sys.stderr)
        except Exception as e:
            print(f"DEBUG: Failed to load model files: {str(e)}", file=sys.stderr)
            clf = None
    
    # Process each row
    for idx, row in df_in.iterrows():
        predicted = None
        confidence = None
        all_probabilities = {}
        proba_array = None
        
        # Try model prediction if available
        if clf is not None and training_features is not None:
            try:
                # Create single-row DataFrame
                row_df = pd.DataFrame([row])
                
                # Remove Tagalog columns (preserve label columns if they exist)
                preserve_set = set()
                if acad_col:
                    preserve_set.add(acad_col)
                if tvl_col:
                    preserve_set.add(tvl_col)
                row_df = remove_tagalog_columns(row_df, preserve_set)
                
                # Prepare features (same as training)
                drop_cols = []
                if acad_col and acad_col in row_df.columns:
                    drop_cols.append(acad_col)
                if tvl_col and tvl_col in row_df.columns:
                    drop_cols.append(tvl_col)
                
                X_row, _, _, _, _, _ = prepare_features(
                    row_df, 
                    drop_cols=drop_cols, 
                    is_training=False, 
                    scaler=scaler, 
                    training_medians=training_medians
                )
                
                # Align with training features
                X_row = X_row.reindex(columns=training_features, fill_value=0)
                X_row = X_row.fillna(0)
                
                # Predict
                if X_row.shape[1] == len(training_features):
                    proba = clf.predict_proba(X_row)[0]
                    proba_array = proba
                    labels = list(getattr(clf, 'classes_', ['Academic', 'TechPro', 'Undecided']))
                    
                    # Create probability dictionary
                    for i, label in enumerate(labels):
                        all_probabilities[label] = float(proba[i])
                    
                    idx_max = int(np.argmax(proba))
                    predicted = labels[idx_max]
                    confidence = float(np.max(proba))
                    
                    # Only keep "Undecided" if probabilities are truly close (within 5%)
                    if predicted == "Undecided":
                        acad_idx = labels.index("Academic") if "Academic" in labels else -1
                        techpro_idx = labels.index("TechPro") if "TechPro" in labels else -1
                        
                        if acad_idx >= 0 and techpro_idx >= 0:
                            acad_prob = proba[acad_idx]
                            techpro_prob = proba[techpro_idx]
                            prob_diff = abs(acad_prob - techpro_prob)
                            
                            # Only keep Undecided if probabilities are very close (within 0.05 = 5%)
                            if prob_diff > 0.05:
                                # Pick the one with higher probability
                                if acad_prob > techpro_prob:
                                    predicted = "Academic"
                                    confidence = float(acad_prob)
                                else:
                                    predicted = "TechPro"
                                    confidence = float(techpro_prob)
                            # If prob_diff <= 0.05, keep "Undecided" (truly neutral)
            except Exception as e:
                print(f"DEBUG: Model prediction failed for row {idx}: {str(e)}", file=sys.stderr)
                predicted = None
        
        # Fallback to simple rule if model failed
        if predicted is None:
            predicted, confidence = simple_rule_prediction(row, acad_col, tvl_col)
            # Create basic probability estimates for fallback
            if predicted == "Academic":
                all_probabilities = {'Academic': confidence, 'TechPro': 1 - confidence, 'Undecided': 0.0}
            elif predicted == "TechPro":
                all_probabilities = {'Academic': 1 - confidence, 'TechPro': confidence, 'Undecided': 0.0}
            else:
                all_probabilities = {'Academic': 0.33, 'TechPro': 0.33, 'Undecided': 0.34}
            proba_array = [all_probabilities.get('Academic', 0), all_probabilities.get('TechPro', 0), all_probabilities.get('Undecided', 0)]
        
        strand_breakdown = generate_strand_breakdown(predicted)
        
        # Get student data using flexible column detection
        student_id = str(pick_case_insensitive(row, ['student_id','student id','id','lrn','studentid']) or '').strip()
        student_name = str(pick_case_insensitive(row, ['student_name','student name','name','full_name','full name']) or '').strip()
        grade_section = str(pick_case_insensitive(row, ['grade_section','grade section','section','grade & section','grade and section']) or '').strip()
        email = pick_case_insensitive(row, ['gmail','email','email address'])
        gender = pick_case_insensitive(row, ['gender','sex'])
        age = pick_case_insensitive(row, ['age'])
        
        # Generate structured recommendation
        structured_rec = None
        try:
            structured_rec = generate_structured_recommendation(
                row, 
                predicted, 
                confidence, 
                all_probabilities if all_probabilities else proba_array,
                list(df_in.columns)
            )
        except Exception as e:
            print(f"DEBUG: Failed to generate structured recommendation for row {idx}: {str(e)}", file=sys.stderr)
        
        result_item = {
            'student_id': student_id,
            'student_name': student_name,
            'gmail': str(email).strip() if email is not None else '',
            'gender': str(gender).strip() if gender is not None else '',
            'age': int(age) if pd.notnull(age) and str(age).strip() != '' else '',
            'grade_section': grade_section,
            'predicted_track': predicted,
            'confidence': round(float(confidence), 2) if confidence is not None else 0.75,
            'strand_breakdown': strand_breakdown
        }
        
        # Add structured recommendation if available
        if structured_rec:
            result_item['recommendation'] = structured_rec
        
        results.append(result_item)
    
    # Ensure excel dir exists
    os.makedirs(EXCEL_DIR, exist_ok=True)
    csv_out = os.path.join(EXCEL_DIR, 'student_predictions.csv')
    xlsx_out = os.path.join(EXCEL_DIR, 'student_predictions.xlsx')
    
    # Save CSV and XLSX (excluding student_name, student_n, gmail, email, and strand_breakdown)
    df_out = pd.DataFrame(results)
    columns_to_drop = ['student_name', 'student_n', 'gmail', 'email', 'strand_breakdown']
    cols_to_drop = []
    for col in df_out.columns:
        col_lower = str(col).lower()
        if any(excluded_col.lower() in col_lower for excluded_col in columns_to_drop):
            cols_to_drop.append(col)
    if cols_to_drop:
        df_out = df_out.drop(columns=cols_to_drop)
    
    df_out.to_csv(csv_out, index=False)
    # Save XLSX
    try:
        df_out.to_excel(xlsx_out, index=False)
    except Exception:
        # openpyxl missing or file locked: ignore silently, CSV still exists
        pass
    
    # Emit per-line JSON for PHP endpoint
    for item in results:
        print(json.dumps(item, ensure_ascii=False))
    
    return 0


def generate_structured_recommendation(row, predicted_track, confidence, all_probabilities, df_columns):
    """
    Generate structured recommendation data matching the UI format.
    Returns a dictionary with recommended track, confidence, reasons, and compatibility scores.
    """
    interest_keywords = ["interested", "enjoy", "like", "learn best"]
    skill_keywords = ["I can", "I am", "I pay attention"]
    track_keywords = ["prefer", "want"]
    grade_keywords = ["Grade 7", "Grade 8", "Grade 9", "Grade 10"]
    
    # Get row index (handle both Series and DataFrame row)
    row_index = row.index if hasattr(row, 'index') else row.keys()
    
    # Detect student's interests, skills, preferences, and grades
    interests = [col for col in df_columns if any(k in col.lower() for k in interest_keywords)]
    skills = [col for col in df_columns if any(k in col.lower() for k in skill_keywords)]
    wants = [col for col in df_columns if any(k in col.lower() for k in track_keywords)]
    grade_cols = [col for col in df_columns if any(k in col for k in grade_keywords)]
    
    # Get strong interests/skills/wants with their values (>=4)
    strong_interests = []
    strong_skills = []
    preferred_tracks = []
    strong_grades = []
    
    for col in interests:
        if col in row_index:
            try:
                val = float(row.get(col, 0) or 0)
                if pd.notnull(val) and val >= 4:
                    strong_interests.append((col, val))
            except (ValueError, TypeError):
                pass
    
    for col in skills:
        if col in row_index:
            try:
                val = float(row.get(col, 0) or 0)
                if pd.notnull(val) and val >= 4:
                    strong_skills.append((col, val))
            except (ValueError, TypeError):
                pass
    
    for col in wants:
        if col in row_index:
            try:
                val = float(row.get(col, 0) or 0)
                if pd.notnull(val) and val >= 4:
                    preferred_tracks.append((col, val))
            except (ValueError, TypeError):
                pass
    
    for col in grade_cols:
        if col in row_index:
            try:
                val = float(row.get(col, 0) or 0)
                if pd.notnull(val) and val >= 3.5:  # Good grades
                    strong_grades.append((col, val))
            except (ValueError, TypeError):
                pass
    
    # Sort by value (highest first)
    strong_interests = sorted(strong_interests, key=lambda x: x[1], reverse=True)[:5]
    strong_skills = sorted(strong_skills, key=lambda x: x[1], reverse=True)[:5]
    preferred_tracks = sorted(preferred_tracks, key=lambda x: x[1], reverse=True)[:3]
    strong_grades = sorted(strong_grades, key=lambda x: x[1], reverse=True)[:4]
    
    # Calculate compatibility scores
    academic_prob = 0
    techpro_prob = 0
    undecided_prob = 0
    
    if isinstance(all_probabilities, dict):
        academic_prob = all_probabilities.get('Academic', 0)
        techpro_prob = all_probabilities.get('TechPro', 0)
        undecided_prob = all_probabilities.get('Undecided', 0)
    elif hasattr(all_probabilities, '__iter__') and not isinstance(all_probabilities, str):
        # Handle array/list format - need to know the order
        # Common order: [Academic, TechPro, Undecided] or based on model classes
        try:
            prob_list = list(all_probabilities)
            if len(prob_list) >= 3:
                # Assume order: Academic, TechPro, Undecided
                academic_prob = float(prob_list[0])
                techpro_prob = float(prob_list[1])
                undecided_prob = float(prob_list[2])
            elif len(prob_list) >= 2:
                academic_prob = float(prob_list[0])
                techpro_prob = float(prob_list[1])
                undecided_prob = 1.0 - academic_prob - techpro_prob
        except (ValueError, TypeError, IndexError):
            # If we can't parse, use confidence to estimate
            if predicted_track == "Academic":
                academic_prob = confidence
                techpro_prob = (1 - confidence) / 2
                undecided_prob = (1 - confidence) / 2
            elif predicted_track == "TechPro":
                techpro_prob = confidence
                academic_prob = (1 - confidence) / 2
                undecided_prob = (1 - confidence) / 2
            else:
                academic_prob = 0.33
                techpro_prob = 0.33
                undecided_prob = 0.34
    else:
        # Fallback: use confidence to estimate
        if predicted_track == "Academic":
            academic_prob = confidence
            techpro_prob = max(0, 1 - confidence - 0.1)
            undecided_prob = min(0.1, 1 - confidence - techpro_prob)
        elif predicted_track == "TechPro":
            techpro_prob = confidence
            academic_prob = max(0, 1 - confidence - 0.1)
            undecided_prob = min(0.1, 1 - confidence - academic_prob)
        else:
            academic_prob = 0.33
            techpro_prob = 0.33
            undecided_prob = 0.34
    
    # Determine alternative track compatibility
    if predicted_track == "Academic":
        alternative_track = "Technical-Professional (TechPro) Track"
        alternative_compatibility = round(techpro_prob * 100, 1)
    elif predicted_track == "TechPro":
        alternative_track = "Academic Track"
        alternative_compatibility = round(academic_prob * 100, 1)
    else:
        # For Undecided, show both
        alternative_track = "Academic Track" if academic_prob > techpro_prob else "Technical-Professional (TechPro) Track"
        alternative_compatibility = round(max(academic_prob, techpro_prob) * 100, 1)
    
    # Generate specific reasons (up to 4)
    reasons = []
    
    if predicted_track == "Academic":
        # Reason 1: Strong academic performance
        if strong_grades:
            math_science_grades = [g for g in strong_grades if any(subj in g[0] for subj in ["Math", "Science", "Mathematics"])]
            if math_science_grades:
                reasons.append(
                    "The student demonstrated strong confidence in core academic subjects, particularly in Math and Science, which are essential foundations for the Academic Track."
                )
            elif strong_grades:
                reasons.append(
                    "The student demonstrated strong performance in academic subjects, showing readiness for the challenges of the Academic Track."
                )
        
        # Reason 2: Academic interests
        if strong_interests:
            academic_interests = [i for i in strong_interests if any(term in i[0].lower() for term in ["research", "analysis", "problem", "study", "learn", "academic"])]
            if academic_interests:
                reasons.append(
                    "The student's expressed interests align with academic-oriented fields, such as research, analysis, and problem-solving, making the Academic Track a natural fit."
                )
        
        # Reason 3: Critical thinking skills
        if strong_skills:
            critical_skills = [s for s in strong_skills if any(term in s[0].lower() for term in ["think", "analyze", "critical", "reason", "solve", "attention"])]
            if critical_skills:
                reasons.append(
                    "The student possesses strong critical thinking and analytical skills, which are highly applicable in the Academic Track's focus areas."
                )
        
        # Reason 4: Cognitive abilities
        if strong_grades or strong_skills:
            reasons.append(
                "The student has shown the ability to perform consistently in cognitively demanding tasks, reinforcing readiness for the challenges of the Academic Track."
            )
        
        # Fallback reasons if not enough specific ones
        if len(reasons) < 4:
            if not strong_grades:
                reasons.append("The student shows potential for academic excellence based on their learning preferences and interests.")
            if len(reasons) < 4:
                reasons.append("The student's profile indicates a strong alignment with academic learning approaches and theoretical studies.")
    
    elif predicted_track == "TechPro":
        # Reason 1: Practical interests
        if strong_interests:
            practical_interests = [i for i in strong_interests if any(term in i[0].lower() for term in ["practical", "hands-on", "technical", "build", "create", "apply"])]
            if practical_interests:
                reasons.append(
                    "The student demonstrated strong interest in practical and hands-on activities, which are central to the Technical-Professional Track."
                )
        
        # Reason 2: Technical skills
        if strong_skills:
            technical_skills = [s for s in strong_skills if any(term in s[0].lower() for term in ["technical", "practical", "hands-on", "apply", "build", "create"])]
            if technical_skills:
                reasons.append(
                    "The student possesses strong technical and practical skills, making them well-suited for the Technical-Professional Track's applied learning approach."
                )
        
        # Reason 3: Career-oriented preferences
        if preferred_tracks:
            reasons.append(
                "The student's track preferences indicate a clear inclination toward technical and vocational education, aligning with the Technical-Professional Track."
            )
        
        # Reason 4: Applied learning readiness
        if strong_skills or strong_interests:
            reasons.append(
                "The student has shown readiness for applied learning and technical competencies, which are essential for success in the Technical-Professional Track."
            )
        
        # Fallback reasons
        if len(reasons) < 4:
            reasons.append("The student's profile shows strong compatibility with technical and vocational education pathways.")
            if len(reasons) < 4:
                reasons.append("The student demonstrates aptitude for practical skill development and technical applications.")
    
    else:  # Undecided
        reasons.append("The student's interests are balanced between academic and technical areas, indicating potential in both tracks.")
        reasons.append("Further exploration of the student's specific interests and career goals would help determine the most suitable track.")
        if strong_interests or strong_skills:
            reasons.append("The student shows diverse interests and skills that could be developed in either track.")
        reasons.append("Career counseling and deeper assessment are recommended to help the student make an informed track selection.")
    
    # Ensure we have at least 4 reasons (pad with general ones if needed)
    while len(reasons) < 4:
        if predicted_track == "Academic":
            reasons.append("The student's overall profile indicates strong compatibility with academic learning environments.")
        elif predicted_track == "TechPro":
            reasons.append("The student's overall profile indicates strong compatibility with technical and vocational learning environments.")
        else:
            reasons.append("The student would benefit from exploring both tracks to make an informed decision.")
    
    # Limit to 4 reasons
    reasons = reasons[:4]
    
    return {
        'recommended_track': predicted_track.upper() + " TRACK" if predicted_track != "Undecided" else "UNDECIDED",
        'predicted_track': predicted_track,
        'confidence_percentage': round(confidence * 100, 1),
        'confidence': confidence,
        'reasons': reasons,
        'alternative_track': alternative_track,
        'alternative_compatibility': alternative_compatibility,
        'academic_compatibility': round(academic_prob * 100, 1),
        'techpro_compatibility': round(techpro_prob * 100, 1),
        'undecided_compatibility': round(undecided_prob * 100, 1),
        'strong_interests': [item[0] for item in strong_interests],
        'strong_skills': [item[0] for item in strong_skills],
        'preferred_tracks': [item[0] for item in preferred_tracks],
        'strong_grades': [item[0] for item in strong_grades]
    }


def generate_counselor_explanation(row, predicted_track, confidence, df_columns):
    """
    Generate detailed counselor-friendly explanation for a prediction.
    Includes compatibility reasons, skills, interests, wants, and track alignment.
    Returns a dictionary with explanation details.
    """
    interest_keywords = ["interested", "enjoy", "like", "learn best"]
    skill_keywords = ["I can", "I am", "I pay attention"]
    track_keywords = ["prefer", "want"]
    
    # Get row index (handle both Series and DataFrame row)
    row_index = row.index if hasattr(row, 'index') else row.keys()
    
    # Detect student's interests and preferences
    interests = [col for col in df_columns if any(k in col.lower() for k in interest_keywords)]
    skills = [col for col in df_columns if any(k in col.lower() for k in skill_keywords)]
    wants = [col for col in df_columns if any(k in col.lower() for k in track_keywords)]
    
    # Get strong interests/skills/wants with their values (>=4)
    strong_interests = []
    strong_skills = []
    preferred_tracks = []
    
    for col in interests:
        if col in row_index:
            try:
                val = float(row.get(col, 0) or 0)
                if pd.notnull(val) and val >= 4:
                    strong_interests.append((col, val))
            except (ValueError, TypeError):
                pass
    
    for col in skills:
        if col in row_index:
            try:
                val = float(row.get(col, 0) or 0)
                if pd.notnull(val) and val >= 4:
                    strong_skills.append((col, val))
            except (ValueError, TypeError):
                pass
    
    for col in wants:
        if col in row_index:
            try:
                val = float(row.get(col, 0) or 0)
                if pd.notnull(val) and val >= 4:
                    preferred_tracks.append((col, val))
            except (ValueError, TypeError):
                pass
    
    # Sort by value (highest first) and take top items
    strong_interests = sorted(strong_interests, key=lambda x: x[1], reverse=True)[:5]
    strong_skills = sorted(strong_skills, key=lambda x: x[1], reverse=True)[:5]
    preferred_tracks = sorted(preferred_tracks, key=lambda x: x[1], reverse=True)[:3]
    
    # Extract just the column names for display
    interest_names = [item[0] for item in strong_interests]
    skill_names = [item[0] for item in strong_skills]
    want_names = [item[0] for item in preferred_tracks]
    
    # Build detailed explanation
    explanation_parts = []
    
    if predicted_track == "Academic":
        explanation_parts.append(f"📘 PREDICTED TRACK: Academic Track")
        explanation_parts.append(f"Confidence Level: {confidence:.1%}")
        explanation_parts.append("")
        explanation_parts.append("🎯 COMPATIBILITY ANALYSIS:")
        explanation_parts.append("The result indicates Academic Track because you demonstrated strong interest and aptitude in academic subjects. Your responses show a preference for theoretical learning, critical thinking, and academic challenges.")
        explanation_parts.append("")
        
        if interest_names:
            explanation_parts.append("💡 IDENTIFIED INTERESTS (Rating ≥ 4):")
            for name, val in strong_interests[:5]:
                explanation_parts.append(f"  • {name} (Rating: {val:.1f})")
        else:
            explanation_parts.append("💡 IDENTIFIED INTERESTS: No high interests found (≥4)")
        
        explanation_parts.append("")
        
        if skill_names:
            explanation_parts.append("🛠️ IDENTIFIED SKILLS (Rating ≥ 4):")
            for name, val in strong_skills[:5]:
                explanation_parts.append(f"  • {name} (Rating: {val:.1f})")
        else:
            explanation_parts.append("🛠️ IDENTIFIED SKILLS: No high skills found (≥4)")
        
        explanation_parts.append("")
        
        if want_names:
            explanation_parts.append("🎓 TRACK PREFERENCES (Rating ≥ 4):")
            for name, val in preferred_tracks[:3]:
                explanation_parts.append(f"  • {name} (Rating: {val:.1f})")
        else:
            explanation_parts.append("🎓 TRACK PREFERENCES: No strong preferences found (≥4)")
        
        explanation_parts.append("")
        explanation_parts.append("✅ COMPATIBILITY WITH ACADEMIC TRACK:")
        explanation_parts.append("Your profile is highly compatible with the Academic Track because:")
        explanation_parts.append("  • High interest in academic learning and theoretical studies")
        explanation_parts.append("  • Strong ability in critical thinking and analysis")
        explanation_parts.append("  • Willingness to engage with challenging academic content")
        explanation_parts.append("  • Preference for research-based and analytical approaches to learning")
        explanation_parts.append("")
        explanation_parts.append("Within the Academic Track, you can explore elective clusters such as:")
        explanation_parts.append("  • Arts, Social Sciences, and Humanities")
        explanation_parts.append("  • Business and Entrepreneurship")
        explanation_parts.append("  • Science, Technology, Engineering, and Mathematics (STEM)")
        explanation_parts.append("  • Sports, Health, and Wellness")
        explanation_parts.append("  • Field Experience")
        
    elif predicted_track == "TechPro":
        explanation_parts.append(f"⚙️ PREDICTED TRACK: Technical-Professional Track")
        explanation_parts.append(f"Confidence Level: {confidence:.1%}")
        explanation_parts.append("")
        explanation_parts.append("🎯 COMPATIBILITY ANALYSIS:")
        explanation_parts.append("The result indicates Technical-Professional Track because you demonstrated interest and aptitude in practical or hands-on activities. Your responses show a preference for applied knowledge, practical skills, and technical work.")
        explanation_parts.append("")
        
        if interest_names:
            explanation_parts.append("💡 IDENTIFIED INTERESTS (Rating ≥ 4):")
            for name, val in strong_interests[:5]:
                explanation_parts.append(f"  • {name} (Rating: {val:.1f})")
        else:
            explanation_parts.append("💡 IDENTIFIED INTERESTS: No high interests found (≥4)")
        
        explanation_parts.append("")
        
        if skill_names:
            explanation_parts.append("🛠️ IDENTIFIED SKILLS (Rating ≥ 4):")
            for name, val in strong_skills[:5]:
                explanation_parts.append(f"  • {name} (Rating: {val:.1f})")
        else:
            explanation_parts.append("🛠️ IDENTIFIED SKILLS: No high skills found (≥4)")
        
        explanation_parts.append("")
        
        if want_names:
            explanation_parts.append("🎓 TRACK PREFERENCES (Rating ≥ 4):")
            for name, val in preferred_tracks[:3]:
                explanation_parts.append(f"  • {name} (Rating: {val:.1f})")
        else:
            explanation_parts.append("🎓 TRACK PREFERENCES: No strong preferences found (≥4)")
        
        explanation_parts.append("")
        explanation_parts.append("✅ COMPATIBILITY WITH TECHNICAL-PROFESSIONAL TRACK:")
        explanation_parts.append("Your profile is highly compatible with the Technical-Professional Track because:")
        explanation_parts.append("  • High interest in practical and hands-on activities")
        explanation_parts.append("  • Strong ability in technical application of knowledge")
        explanation_parts.append("  • Willingness to learn practical skills for employment")
        explanation_parts.append("  • Preference for applied learning and technical competencies")
        explanation_parts.append("")
        explanation_parts.append("Within the Technical-Professional Track, you can explore elective clusters such as:")
        explanation_parts.append("  • Agriculture and Fishery Arts")
        explanation_parts.append("  • Information and Communication Technology")
        explanation_parts.append("  • Family and Consumer Science")
        explanation_parts.append("  • Industrial Arts and Maritime")
        explanation_parts.append("  • Field Experience")
        
    else:  # Undecided
        explanation_parts.append(f"🤔 PREDICTED TRACK: Undecided")
        explanation_parts.append(f"Confidence Level: {confidence:.1%}")
        explanation_parts.append("")
        explanation_parts.append("🎯 COMPATIBILITY ANALYSIS:")
        explanation_parts.append("The result indicates Undecided because your interests are balanced between academic and technical areas. You show potential in both tracks and may benefit from deeper exploration of your interests and abilities.")
        explanation_parts.append("")
        
        if interest_names:
            explanation_parts.append("💡 IDENTIFIED INTERESTS (Rating ≥ 4):")
            for name, val in strong_interests[:5]:
                explanation_parts.append(f"  • {name} (Rating: {val:.1f})")
        else:
            explanation_parts.append("💡 IDENTIFIED INTERESTS: No high interests found (≥4)")
        
        explanation_parts.append("")
        
        if skill_names:
            explanation_parts.append("🛠️ IDENTIFIED SKILLS (Rating ≥ 4):")
            for name, val in strong_skills[:5]:
                explanation_parts.append(f"  • {name} (Rating: {val:.1f})")
        else:
            explanation_parts.append("🛠️ IDENTIFIED SKILLS: No high skills found (≥4)")
        
        explanation_parts.append("")
        
        if want_names:
            explanation_parts.append("🎓 TRACK PREFERENCES (Rating ≥ 4):")
            for name, val in preferred_tracks[:3]:
                explanation_parts.append(f"  • {name} (Rating: {val:.1f})")
        else:
            explanation_parts.append("🎓 TRACK PREFERENCES: No strong preferences found (≥4)")
        
        explanation_parts.append("")
        explanation_parts.append("💭 RECOMMENDATION:")
        explanation_parts.append("Consider reflecting on whether you prefer:")
        explanation_parts.append("  • Academic learning (theoretical, research-based approaches)")
        explanation_parts.append("  • Practical skills (hands-on, technical work applications)")
        explanation_parts.append("")
        explanation_parts.append("Career counseling and deeper assessment may help you choose the appropriate track and elective clusters that align with your interests, abilities, and career goals.")
    
    explanation = "\n".join(explanation_parts)
    
    return {
        'explanation': explanation,
        'strong_interests': [item[0] for item in strong_interests],
        'interest_scores': {item[0]: item[1] for item in strong_interests},
        'strong_skills': [item[0] for item in strong_skills],
        'skill_scores': {item[0]: item[1] for item in strong_skills},
        'preferred_tracks': [item[0] for item in preferred_tracks],
        'preference_scores': {item[0]: item[1] for item in preferred_tracks},
        'predicted_track': predicted_track,
        'confidence': confidence
    }


def run_training_mode(test_csv_path=None, generate_explanations=False):
    if not ML_AVAILABLE:
        print(json.dumps({'error': 'Machine learning libraries not available. Install: pip install scikit-learn imbalanced-learn joblib'}))
        return 1
    
    print("="*80, file=sys.stderr)
    print("📘 StrandAlign ML Training Pipeline", file=sys.stderr)
    print("="*80, file=sys.stderr)
    
    training_csv = os.path.join(SCRIPT_DIR, 'Student_prediction.csv')
    if not os.path.exists(training_csv):
        print(json.dumps({
            'error': 'Training CSV not found',
            'expected_path': training_csv
        }))
        return 1
    
    print(f"\n📂 Loading CSV: {training_csv}", file=sys.stderr)
    try:
        df = pd.read_csv(training_csv, encoding='utf-8')
    except Exception:
        try:
            df = pd.read_csv(training_csv, encoding='latin1')
        except Exception as e:
            print(json.dumps({'error': f'Failed to read training CSV: {str(e)}'}))
            return 1
    
    print(f"Raw shape: {df.shape}", file=sys.stderr)
    
    # Find label columns before cleaning
    academic_candidates, techpro_candidates = find_label_columns(df)
    
    # Broader search if not found
    if not academic_candidates or not techpro_candidates:
        cols = list(df.columns)
        low = [c.lower() for c in cols]
        if not academic_candidates:
            academic_candidates = [cols[i] for i, c in enumerate(low) 
                                  if ("academic" in c and "track" in c) or ("academic track" in c) 
                                  or ("academic" in c and ("stem" in c or "abm" in c))]
        if not techpro_candidates:
            techpro_candidates = [cols[i] for i, c in enumerate(low) 
                                 if "technical" in c or "technical-professional" in c 
                                 or "technical professional" in c or ("ict" in c and "technical" in c)]
    
    if not academic_candidates or not techpro_candidates:
        print(json.dumps({
            'error': 'Could not detect Academic and Technical-Professional Track columns',
            'available_columns': list(df.columns)[:40]
        }))
        return 1
    
    academic_col = academic_candidates[0]
    techpro_col = techpro_candidates[0]
    print(f"✅ Detected label columns (preserving these during cleaning):\n - Academic: {academic_col}\n - TechPro: {techpro_col}\n", file=sys.stderr)
    
    # Remove Tagalog columns (preserve label columns)
    print("🧹 Cleaning bilingual duplicates (keeping English only)...", file=sys.stderr)
    initial_cols = df.shape[1]
    df = remove_tagalog_columns(df, preserve_cols={academic_col, techpro_col})
    removed = initial_cols - df.shape[1]
    print(f"✅ Removed {removed} Tagalog duplicate columns.", file=sys.stderr)
    print(f"Remaining columns: {df.shape[1]}\n", file=sys.stderr)
    
    # Re-detect label columns after cleaning
    academic_col_list = [c for c in df.columns if "Academic Track" in c or academic_col in c]
    techpro_col_list = [c for c in df.columns if "Technical-Professional Track" in c or "Technical" in c or techpro_col in c]
    
    if not academic_col_list or not techpro_col_list:
        print(json.dumps({'error': 'Label columns not found after cleaning'}))
        return 1
    
    academic_col = academic_col_list[0]
    techpro_col = techpro_col_list[0]
    
    # Derive labels - using >=3 threshold like user's script
    def derive_group_label(row):
        a = row[academic_col]
        t = row[techpro_col]
        if a >= 3 and t < 3:
            return "Academic"
        elif t >= 3 and a < 3:
            return "TechPro"
        elif t >= 3 and a >= 3:
            return "Undecided"
        else:
            return "Undecided"
    
    df['derived_group_label'] = df.apply(derive_group_label, axis=1)
    print("🎯 Derived group label distribution:", file=sys.stderr)
    print(df['derived_group_label'].value_counts(), file=sys.stderr)
    print("", file=sys.stderr)
    
    # Prepare features
    drop_cols = [academic_col, techpro_col, 'derived_group_label']
    X, grade_cols, preference_cols, scaler, feature_names, training_medians = prepare_features(
        df, drop_cols=drop_cols, is_training=True
    )
    y = df['derived_group_label']
    
    if grade_cols:
        print("🧮 Normalizing grade columns...", file=sys.stderr)
    print(f"Including {len(feature_names)} features (with preferences) for training.", file=sys.stderr)
    print(f"Sample columns: {feature_names[:10]}\n", file=sys.stderr)
    
    # Cross-validation with enhanced metrics
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_accuracies, fold_f1s, fold_precisions, fold_recalls = [], [], [], []
    best_params_global = None
    
    for fold_num, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
        print("="*80, file=sys.stderr)
        print(f"📊 Fold {fold_num}/{N_SPLITS}", file=sys.stderr)
        print("="*80, file=sys.stderr)
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        class_counts = Counter(y_train)
        print(f"📈 Original class distribution: {dict(class_counts)}", file=sys.stderr)
        
        # SMOTE with dynamic k_neighbors
        min_class_size = min(class_counts.values())
        if min_class_size < 2:
            print("⚠️ Skipping SMOTE (minority class <2).", file=sys.stderr)
            X_train_bal, y_train_bal = X_train, y_train
        else:
            k_value = min(5, max(1, min_class_size - 1))
            print(f"⚙️ Applying SMOTE with k_neighbors={k_value}", file=sys.stderr)
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_value)
            X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
            print(f"✅ Balanced class distribution: {dict(Counter(y_train_bal))}", file=sys.stderr)
        
        # Hyperparameter search
        clf = RandomForestClassifier(random_state=RANDOM_STATE)
        param_grid = {
            'n_estimators': [200, 400, 600],
            'max_depth': [20, 40, 60, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        search = RandomizedSearchCV(
            clf, param_grid, n_iter=20, scoring='f1_weighted', cv=3,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=0
        )
        search.fit(X_train_bal, y_train_bal)
        best_clf = search.best_estimator_
        if best_params_global is None:
            best_params_global = search.best_params_
        
        y_pred = best_clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        fold_accuracies.append(acc)
        fold_f1s.append(f1)
        fold_precisions.append(prec)
        fold_recalls.append(rec)
        
        print(f"🏁 Best Params (Fold {fold_num}): {search.best_params_}", file=sys.stderr)
        print(f"🎯 Accuracy: {acc:.4f} | F1-weighted: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}", file=sys.stderr)
        print("Confusion Matrix:", file=sys.stderr)
        print(confusion_matrix(y_test, y_pred), file=sys.stderr)
    
    # Cross-validation summary
    mean_acc, std_acc = np.mean(fold_accuracies), np.std(fold_accuracies)
    mean_f1, std_f1 = np.mean(fold_f1s), np.std(fold_f1s)
    mean_prec, std_prec = np.mean(fold_precisions), np.std(fold_precisions)
    mean_rec, std_rec = np.mean(fold_recalls), np.std(fold_recalls)
    
    print("\n" + "="*80, file=sys.stderr)
    print("📈 Stratified K-Fold Summary", file=sys.stderr)
    print("="*80, file=sys.stderr)
    print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}", file=sys.stderr)
    print(f"Mean F1-weighted: {mean_f1:.4f} ± {std_f1:.4f}", file=sys.stderr)
    print(f"Mean Precision: {mean_prec:.4f} ± {std_prec:.4f}", file=sys.stderr)
    print(f"Mean Recall: {mean_rec:.4f} ± {std_rec:.4f}", file=sys.stderr)
    
    # Final model training with best params
    print("\n🔧 Training final calibrated model...", file=sys.stderr)
    min_class_size_full = min(Counter(y).values())
    if min_class_size_full < 2:
        print("⚠️ Not enough samples for SMOTE. Proceeding without balancing.", file=sys.stderr)
        X_bal, y_bal = X, y
    else:
        k_value_full = min(5, max(1, min_class_size_full - 1))
        smote_full = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_value_full)
        X_bal, y_bal = smote_full.fit_resample(X, y)
    
    print(f"✅ Balanced full dataset distribution: {dict(Counter(y_bal))}", file=sys.stderr)
    
    # Use best params from search, or defaults if search didn't run
    best_params = best_params_global if best_params_global is not None else {
        'n_estimators': 400,
        'max_depth': 40,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'bootstrap': True,
        'class_weight': 'balanced_subsample'
    }
    
    base_rf = RandomForestClassifier(**best_params, random_state=RANDOM_STATE)
    calibrated_rf = CalibratedClassifierCV(base_rf, method='isotonic', cv=3)
    calibrated_rf.fit(X_bal, y_bal)
    
    # Feature importance extraction
    print("\n🔥 Extracting feature importances...", file=sys.stderr)
    importances = np.mean(
        [cc.estimator.feature_importances_ for cc in calibrated_rf.calibrated_classifiers_],
        axis=0
    )
    
    feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance (%)": importances * 100
    }).sort_values(by="Importance (%)", ascending=False)
    
    print("Top 10 Most Influential Features:", file=sys.stderr)
    for i in range(min(10, len(feat_imp))):
        print(f"  {feat_imp.iloc[i, 0]}: {feat_imp.iloc[i, 1]:.2f}%", file=sys.stderr)
    
    feat_imp.to_csv(FEATURE_IMPORTANCE_OUTPUT, index=False)
    print(f"💾 Saved feature importances to {FEATURE_IMPORTANCE_OUTPUT}", file=sys.stderr)
    
    # Save model, scaler, and feature names
    joblib.dump(calibrated_rf, MODEL_OUTPUT)
    joblib.dump(scaler, SCALER_OUTPUT)
    
    feature_data = {
        'feature_names': feature_names,
        'medians': training_medians.to_dict() if training_medians is not None else {}
    }
    with open(FEATURES_OUTPUT, 'w') as f:
        json.dump(feature_data, f, indent=2)
    
    print(f"\n✅ Model saved: {MODEL_OUTPUT}", file=sys.stderr)
    print(f"✅ Scaler saved: {SCALER_OUTPUT}", file=sys.stderr)
    print(f"✅ Features saved: {FEATURES_OUTPUT}", file=sys.stderr)
    
    # Test on unseen data if provided
    if test_csv_path and os.path.exists(test_csv_path):
        print(f"\n🔍 Testing on unseen data: {test_csv_path}", file=sys.stderr)
        try:
            df_test = pd.read_csv(test_csv_path, encoding='utf-8')
        except Exception:
            try:
                df_test = pd.read_csv(test_csv_path, encoding='latin1')
            except Exception as e:
                print(f"⚠️ Failed to read test CSV: {str(e)}", file=sys.stderr)
                df_test = None
        
        if df_test is not None:
            # Keep original test data for explanations (before feature alignment)
            df_test_original = df_test.copy()
            
            # Align, fill missing, and scale grade columns for prediction
            df_test_features = df_test.reindex(columns=feature_names, fill_value=np.nan)
            for col in df_test_features.columns:
                if col in feature_names:
                    df_test_features[col] = pd.to_numeric(df_test_features[col], errors='coerce')
                    if col in training_medians:
                        df_test_features[col] = df_test_features[col].fillna(training_medians[col])
                    else:
                        df_test_features[col] = df_test_features[col].fillna(0)
            
            if grade_cols:
                available_grade_cols = [c for c in grade_cols if c in df_test_features.columns]
                if available_grade_cols:
                    df_test_features[available_grade_cols] = scaler.transform(df_test_features[available_grade_cols])
            
            y_pred_test = calibrated_rf.predict(df_test_features)
            y_pred_prob = calibrated_rf.predict_proba(df_test_features)
            
            test_predictions = pd.DataFrame({
                "Prediction": y_pred_test,
                "Confidence": np.max(y_pred_prob, axis=1)
            })
            
            test_output = os.path.join(EXCEL_DIR, 'test_predictions.csv')
            os.makedirs(EXCEL_DIR, exist_ok=True)
            test_predictions.to_csv(test_output, index=False)
            print(f"💾 Saved test predictions to {test_output}", file=sys.stderr)
            
            # Generate counselor explanations if requested
            if generate_explanations:
                print("\n🧭 Generating counselor-friendly explanations...", file=sys.stderr)
                print("", file=sys.stderr)
                
                # Combine original data with predictions
                df_students = df_test_original.copy()
                df_students["Predicted_Track"] = y_pred_test
                df_students["Confidence"] = np.max(y_pred_prob, axis=1)
                
                explanations = []
                all_explanation_data = []
                
                for idx, row in df_students.iterrows():
                    pred = row["Predicted_Track"]
                    conf = float(row["Confidence"])
                    
                    # Use all original columns for explanation (not just features)
                    expl = generate_counselor_explanation(row, pred, conf, list(df_students.columns))
                    
                    explanations.append({
                        'Predicted_Track': pred,
                        'Confidence': conf,
                        'Counselor_Explanation': expl['explanation']
                    })
                    
                    # Store detailed data
                    all_explanation_data.append({
                        'Predicted_Track': pred,
                        'Confidence': conf,
                        'Counselor_Explanation': expl['explanation'],
                        'Strong_Interests': ', '.join(expl['strong_interests'][:5]) if expl['strong_interests'] else 'N/A',
                        'Strong_Skills': ', '.join(expl['strong_skills'][:5]) if expl['strong_skills'] else 'N/A',
                        'Preferred_Tracks': ', '.join(expl['preferred_tracks'][:3]) if expl['preferred_tracks'] else 'N/A'
                    })
                
                # Save detailed explanations
                expl_df = pd.DataFrame(all_explanation_data)
                expl_output = os.path.join(EXCEL_DIR, 'counselor_explanations.csv')
                expl_df.to_csv(expl_output, index=False)
                print(f"💾 Saved explanations to {expl_output}", file=sys.stderr)
                
                # Print sample explanations
                print("\n🎓 Sample Explanations:", file=sys.stderr)
                for i in range(min(3, len(explanations))):
                    print("\n" + "-"*60, file=sys.stderr)
                    print(explanations[i]['Counselor_Explanation'], file=sys.stderr)
                
                print("\n✅ Guidance counselor explanations generated successfully.", file=sys.stderr)
    
    # Output JSON summary
    summary_metrics = {
        'model_saved': MODEL_OUTPUT,
        'scaler_saved': SCALER_OUTPUT,
        'features_saved': FEATURES_OUTPUT,
        'feature_importance_saved': FEATURE_IMPORTANCE_OUTPUT,
        'mean_accuracy': round(mean_acc, 4),
        'std_accuracy': round(std_acc, 4),
        'mean_f1_weighted': round(mean_f1, 4),
        'std_f1_weighted': round(std_f1, 4),
        'mean_precision': round(mean_prec, 4),
        'std_precision': round(std_prec, 4),
        'mean_recall': round(mean_rec, 4),
        'std_recall': round(std_rec, 4),
        'num_features': len(feature_names),
        'best_params': best_params
    }
    
    print("\n📊 Model Performance Summary (JSON):", file=sys.stderr)
    print(json.dumps(summary_metrics, indent=2), file=sys.stderr)
    print(json.dumps(summary_metrics, indent=2))
    
    print("\n✅ Calibrated StrandAlign Model Training Completed Successfully.", file=sys.stderr)
    return 0


if __name__ == '__main__':
    # If an argument is provided, run prediction mode; otherwise training mode
    if len(sys.argv) >= 2 and os.path.exists(sys.argv[1]):
        print(f"Starting prediction mode with file: {sys.argv[1]}", file=sys.stderr)
        sys.exit(run_prediction_mode(sys.argv[1]))
    else:
        if ML_AVAILABLE:
            # Optional: support test CSV and explanations via environment variables or additional args
            test_csv = None
            generate_expl = False
            if len(sys.argv) >= 3:
                test_csv = sys.argv[2] if os.path.exists(sys.argv[2]) else None
            if len(sys.argv) >= 4:
                generate_expl = sys.argv[3].lower() in ['true', '1', 'yes']
            sys.exit(run_training_mode(test_csv_path=test_csv, generate_explanations=generate_expl))
        else:
            print(json.dumps({'error': 'Training mode requires ML libraries. Use prediction mode with CSV file.'}))
            sys.exit(1)
