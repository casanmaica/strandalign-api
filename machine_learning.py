import pandas as pd
import numpy as np
import json
import sys
import os
import pickle

# Optional imports for training mode
try:
    from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, classification_report
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.calibration import CalibratedClassifierCV
    from imblearn.over_sampling import SMOTE
    from collections import Counter
    import joblib
    import matplotlib
    # Try to use an interactive backend for on-screen display
    # Falls back to 'Agg' if interactive backend is not available
    try:
        # Try interactive backends in order of preference
        matplotlib.use('TkAgg')  # Try TkAgg first (works on Windows, Linux, Mac)
    except:
        try:
            matplotlib.use('Qt5Agg')  # Try Qt5Agg as fallback
        except:
            matplotlib.use('Agg')  # Fallback to non-interactive if needed
    import matplotlib.pyplot as plt
    import seaborn as sns
    ML_AVAILABLE = True
    VISUALIZATION_AVAILABLE = True
    SHOW_GRAPHS = True  # Enable on-screen display
except ImportError:
    ML_AVAILABLE = False
    VISUALIZATION_AVAILABLE = False
    SHOW_GRAPHS = False
    try:
        import matplotlib
        try:
            matplotlib.use('TkAgg')
        except:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        VISUALIZATION_AVAILABLE = True
        SHOW_GRAPHS = True
    except ImportError:
        VISUALIZATION_AVAILABLE = False
        SHOW_GRAPHS = False

# Paths relative to this script
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback when __file__ is not defined
    SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv else os.getcwd()

EXCEL_DIR = os.path.join(SCRIPT_DIR, 'excel')
VISUALIZATIONS_DIR = os.path.join(SCRIPT_DIR, 'visualizations')
MODEL_OUTPUT = os.path.join(SCRIPT_DIR, 'strandalign_model.joblib')
SCALER_OUTPUT = os.path.join(SCRIPT_DIR, 'strandalign_scaler.joblib')
FEATURES_OUTPUT = os.path.join(SCRIPT_DIR, 'strandalign_features.json')
FEATURE_IMPORTANCE_OUTPUT = os.path.join(SCRIPT_DIR, 'feature_importances.csv')
RANDOM_STATE = 42
N_SPLITS = 3  # Reduced from 5 to 3 for faster training (still statistically valid)
# Performance improvement settings
USE_ADVANCED_FEATURES = True  # Enable advanced feature engineering
USE_FEATURE_SELECTION = True  # Enable feature selection
FEATURE_SELECTION_METHOD = 'importance'  # 'importance' or 'kbest'
FEATURE_SELECTION_THRESHOLD = 0.003  # Lower threshold for more features (was 0.01)
K_BEST_FEATURES = 25  # For k-best selection (if method is 'kbest')
HYPERPARAM_SEARCH_ITERATIONS = 150  # Increased for better tuning (was 100)

# Speed optimization settings
QUICK_TRAINING_MODE = False  # Set to True for faster training (reduces iterations, folds, etc.)
SAVE_AS_PKL = True  # Save model as .pkl file in addition to .joblib
CV_N_ESTIMATORS = 200  # Use fewer trees during CV for speed (final model uses full n_estimators)
SKIP_HYPERPARAM_SEARCH_IF_MODEL_EXISTS = True  # Skip hyperparameter search if model already exists (use previous best params)

# Academic class performance improvement settings
USE_CLASS_WEIGHT_ADJUSTMENT = True  # Adjust class weights to improve Academic class performance
AUGMENT_ACADEMIC_DATA = True  # Generate synthetic Academic examples to balance dataset
USE_ENHANCED_FEATURES = True  # Add features that better distinguish Academic vs TechPro
ENABLE_DATA_QUALITY_CHECKS = True  # Perform data quality improvements (outlier detection, cleaning)
ACADEMIC_AUGMENTATION_FACTOR = 1.5  # More aggressive: 30% more Academic samples (was 1.2)
# Note: If BALANCE_ALL_CLASSES_EQUALLY is True, this factor is ignored and calculated automatically

# TechPro class performance improvement settings
AUGMENT_TECHPRO_DATA = True  # Generate synthetic TechPro examples to balance dataset
TECHPRO_AUGMENTATION_FACTOR = 1.5  # More aggressive: 30% more TechPro samples (was 1.2)
# Note: If BALANCE_ALL_CLASSES_EQUALLY is True, this factor is ignored and calculated automatically
BOOST_TECHPRO_WEIGHT = True  # Boost TechPro class weight to improve its performance
TECHPRO_WEIGHT_BOOST = 1.15  # 15% boost for TechPro class weight

# Class balancing settings
BALANCE_ALL_CLASSES_EQUALLY = True  # If True, augment all classes to match the largest class size
AUGMENT_UNDECIDED_DATA = True  # Also augment Undecided class if balancing equally


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


def augment_class_data(df, class_mask, class_name, augmentation_factor=1.2, enabled=True):
    """
    Generate synthetic examples for a specific class using SMOTE-like approach with controlled noise.
    
    Args:
        df: DataFrame with data
        class_mask: Boolean mask for the class to augment
        class_name: Name of the class (for logging)
        augmentation_factor: Factor to multiply samples by (1.2 = 20% more)
        enabled: Whether augmentation is enabled for this class
    """
    if not enabled or augmentation_factor <= 1.0:
        return df
    
    class_df = df[class_mask].copy()
    if len(class_df) < 2:
        return df
    
    # Calculate how many samples to generate
    current_count = len(class_df)
    target_count = int(current_count * augmentation_factor)
    samples_needed = max(0, target_count - current_count)
    
    if samples_needed == 0:
        return df
    
    # Generate synthetic samples by adding small random noise
    np.random.seed(RANDOM_STATE)
    synthetic_samples = []
    
    for _ in range(samples_needed):
        # Pick a random sample from the class
        base_idx = np.random.randint(0, len(class_df))
        base_sample = class_df.iloc[base_idx].copy()
        
        # Add controlled noise to numeric columns only
        numeric_cols = class_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in base_sample.index:
                val = base_sample[col]
                if pd.notnull(val) and isinstance(val, (int, float)):
                    # Add small noise (5% of value or 0.1, whichever is smaller)
                    noise = np.random.normal(0, min(abs(val) * 0.05, 0.1))
                    base_sample[col] = max(1, min(5, val + noise))  # Keep in valid range
        
        synthetic_samples.append(base_sample)
    
    if synthetic_samples:
        synthetic_df = pd.DataFrame(synthetic_samples)
        augmented_df = pd.concat([df, synthetic_df], ignore_index=True)
        new_count = len(augmented_df) - len(df) + current_count
        print(f"✅ Augmented {class_name} class: {current_count} → {new_count} samples (+{new_count - current_count})", file=sys.stderr)
        return augmented_df
    
    return df


def augment_academic_data(df, academic_mask, augmentation_factor=1.2):
    """
    Generate synthetic Academic examples using SMOTE-like approach with controlled noise.
    (Wrapper for backward compatibility)
    """
    return augment_class_data(df, academic_mask, "Academic", augmentation_factor, AUGMENT_ACADEMIC_DATA)


def improve_data_quality(df, numeric_cols):
    """
    Perform data quality improvements: outlier detection and cleaning.
    """
    if not ENABLE_DATA_QUALITY_CHECKS:
        return df
    
    df_cleaned = df.copy()
    outliers_removed = 0
    
    for col in numeric_cols:
        if col not in df_cleaned.columns:
            continue
        
        # Use IQR method for outlier detection
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR > 0:
            lower_bound = Q1 - 2.5 * IQR  # More lenient (2.5 instead of 1.5)
            upper_bound = Q3 + 2.5 * IQR
            
            # Cap outliers instead of removing (less data loss)
            before = (df_cleaned[col] < lower_bound).sum() + (df_cleaned[col] > upper_bound).sum()
            if before > 0:
                # Convert to float to avoid dtype warnings
                df_cleaned[col] = df_cleaned[col].astype(float)
                df_cleaned.loc[df_cleaned[col] < lower_bound, col] = float(lower_bound)
                df_cleaned.loc[df_cleaned[col] > upper_bound, col] = float(upper_bound)
            outliers_removed += before
    
    if outliers_removed > 0:
        print(f"✅ Data quality: Capped {outliers_removed} outlier values", file=sys.stderr)
    
    return df_cleaned


def add_enhanced_features(X, grade_cols, subject_interest_cols, ability_cols):
    """
    Add features that better distinguish Academic vs TechPro tracks.
    """
    if not USE_ENHANCED_FEATURES:
        return X
    
    enhanced_X = X.copy()
    
    # Academic vs TechPro distinguishing features
    if grade_cols:
        available_grades = [c for c in grade_cols if c in X.columns]
        if available_grades:
            # Academic strength indicators
            math_science_cols = [c for c in available_grades if any(subj in c for subj in ['Math', 'Science', 'Mathematics'])]
            if math_science_cols:
                enhanced_X['academic_strength_score'] = X[math_science_cols].mean(axis=1)
                enhanced_X['academic_consistency'] = 1.0 / (X[math_science_cols].std(axis=1) + 1e-6)  # Higher = more consistent
            
            # TechPro strength indicators
            tle_cols = [c for c in available_grades if 'TLE' in c]
            if tle_cols:
                enhanced_X['techpro_strength_score'] = X[tle_cols].mean(axis=1)
            
            # Academic vs TechPro performance gap
            if 'academic_strength_score' in enhanced_X.columns and 'techpro_strength_score' in enhanced_X.columns:
                enhanced_X['academic_techpro_gap'] = enhanced_X['academic_strength_score'] - enhanced_X['techpro_strength_score']
            
            # Theoretical vs Practical preference
            theoretical_subjects = [c for c in available_grades if any(subj in c for subj in ['Math', 'Science', 'English', 'Araling Panlipunan'])]
            practical_subjects = [c for c in available_grades if any(subj in c for subj in ['TLE', 'MAPEH'])]
            
            if theoretical_subjects and practical_subjects:
                enhanced_X['theoretical_avg'] = X[theoretical_subjects].mean(axis=1)
                enhanced_X['practical_avg'] = X[practical_subjects].mean(axis=1)
                enhanced_X['theoretical_practical_ratio'] = enhanced_X['theoretical_avg'] / (enhanced_X['practical_avg'] + 1e-6)
    
    # Interest-based distinguishing features
    if subject_interest_cols:
        available_interest = [c for c in subject_interest_cols if c in X.columns]
        if available_interest:
            # Academic interest indicators
            academic_interest_keywords = ['research', 'analysis', 'study', 'learn', 'academic', 'theory', 'problem']
            academic_interests = [c for c in available_interest if any(kw in c.lower() for kw in academic_interest_keywords)]
            
            # TechPro interest indicators
            techpro_interest_keywords = ['practical', 'hands-on', 'technical', 'build', 'create', 'apply', 'skill']
            techpro_interests = [c for c in available_interest if any(kw in c.lower() for kw in techpro_interest_keywords)]
            
            if academic_interests:
                enhanced_X['academic_interest_score'] = X[academic_interests].mean(axis=1)
            else:
                enhanced_X['academic_interest_score'] = 0
            if techpro_interests:
                enhanced_X['techpro_interest_score'] = X[techpro_interests].mean(axis=1)
            else:
                enhanced_X['techpro_interest_score'] = 0
            
            if 'academic_interest_score' in enhanced_X.columns and 'techpro_interest_score' in enhanced_X.columns:
                enhanced_X['interest_alignment_score'] = enhanced_X['academic_interest_score'] - enhanced_X['techpro_interest_score']
    
    # Ability-based distinguishing features
    if ability_cols:
        available_ability = [c for c in ability_cols if c in X.columns]
        if available_ability:
            # Academic ability indicators
            academic_ability_keywords = ['think', 'analyze', 'critical', 'reason', 'solve', 'understand', 'conceptual']
            academic_abilities = [c for c in available_ability if any(kw in c.lower() for kw in academic_ability_keywords)]
            
            # TechPro ability indicators
            techpro_ability_keywords = ['practical', 'hands-on', 'technical', 'apply', 'build', 'create', 'manual']
            techpro_abilities = [c for c in available_ability if any(kw in c.lower() for kw in techpro_ability_keywords)]
            
            if academic_abilities:
                enhanced_X['academic_ability_score'] = X[academic_abilities].mean(axis=1)
            else:
                enhanced_X['academic_ability_score'] = 0
            if techpro_abilities:
                enhanced_X['techpro_ability_score'] = X[techpro_abilities].mean(axis=1)
            else:
                enhanced_X['techpro_ability_score'] = 0
            
            if 'academic_ability_score' in enhanced_X.columns and 'techpro_ability_score' in enhanced_X.columns:
                enhanced_X['ability_alignment_score'] = enhanced_X['academic_ability_score'] - enhanced_X['techpro_ability_score']
    
    # Combined alignment score (most important distinguishing feature)
    alignment_features = []
    if 'academic_techpro_gap' in enhanced_X.columns:
        alignment_features.append('academic_techpro_gap')
    if 'interest_alignment_score' in enhanced_X.columns:
        alignment_features.append('interest_alignment_score')
    if 'ability_alignment_score' in enhanced_X.columns:
        alignment_features.append('ability_alignment_score')
    
    if alignment_features:
        enhanced_X['overall_academic_techpro_alignment'] = enhanced_X[alignment_features].mean(axis=1)
    
    return enhanced_X


def prepare_features(df, drop_cols=None, is_training=True, scaler=None, training_medians=None, feature_selector=None):
    """
    Prepare features with advanced engineering, binarized preferences, and normalization.
    Returns: (X, grade_cols, preference_cols, scaler, feature_names, training_medians, feature_selector)
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
    
    # Advanced feature engineering
    if USE_ADVANCED_FEATURES:
        # Grade statistics
        if grade_cols:
            available_grades = [c for c in grade_cols if c in X.columns]
            if available_grades:
                X['grade_mean'] = X[available_grades].mean(axis=1)
                X['grade_std'] = X[available_grades].std(axis=1).fillna(0)
                X['grade_max'] = X[available_grades].max(axis=1)
                X['grade_min'] = X[available_grades].min(axis=1)
                # Grade trend (improvement over time)
                if len(available_grades) >= 2:
                    # Try to get grades in order (Grade 7, 8, 9, 10)
                    sorted_grades = sorted(available_grades, key=lambda x: ('7' in x, '8' in x, '9' in x, '10' in x))
                    if len(sorted_grades) >= 2:
                        X['grade_trend'] = X[sorted_grades[-1]] - X[sorted_grades[0]]
                    else:
                        X['grade_trend'] = 0
                else:
                    X['grade_trend'] = 0
                
                # Subject-specific averages
                math_grades = [c for c in available_grades if 'Math' in c]
                science_grades = [c for c in available_grades if 'Science' in c]
                tle_grades = [c for c in available_grades if 'TLE' in c]
                english_grades = [c for c in available_grades if 'English' in c]
                
                if math_grades:
                    X['math_avg'] = X[math_grades].mean(axis=1)
                if science_grades:
                    X['science_avg'] = X[science_grades].mean(axis=1)
                if tle_grades:
                    X['tle_avg'] = X[tle_grades].mean(axis=1)
                if english_grades:
                    X['english_avg'] = X[english_grades].mean(axis=1)
                
                # Performance ratios (key differentiators)
                if 'tle_avg' in X.columns and 'math_avg' in X.columns:
                    X['tle_vs_math_ratio'] = X['tle_avg'] / (X['math_avg'] + 1e-6)  # Avoid division by zero
                if 'tle_avg' in X.columns and 'science_avg' in X.columns:
                    practical = X['tle_avg']
                    theoretical = (X['math_avg'] if 'math_avg' in X.columns else 0) + (X['science_avg'] if 'science_avg' in X.columns else 0)
                    X['practical_vs_theoretical'] = practical / (theoretical / 2 + 1e-6)
                if 'grade_max' in X.columns and 'grade_min' in X.columns:
                    X['best_vs_worst_ratio'] = X['grade_max'] / (X['grade_min'] + 1e-6)
        
        # Interest/ability aggregations
        if subject_interest_cols:
            available_interest = [c for c in subject_interest_cols if c in X.columns]
            if available_interest:
                X['interest_mean'] = X[available_interest].mean(axis=1)
                X['interest_std'] = X[available_interest].std(axis=1).fillna(0)
                X['high_interest_count'] = (X[available_interest] >= 4).sum(axis=1)
        
        if ability_cols:
            available_ability = [c for c in ability_cols if c in X.columns]
            if available_ability:
                X['ability_mean'] = X[available_ability].mean(axis=1)
                X['ability_std'] = X[available_ability].std(axis=1).fillna(0)
                X['high_ability_count'] = (X[available_ability] >= 4).sum(axis=1)
        
        # Feature interactions (key combinations)
        if 'grade_mean' in X.columns and 'interest_mean' in X.columns:
            X['grade_interest_alignment'] = X['grade_mean'] * X['interest_mean']
        
        if 'tle_avg' in X.columns and 'interest_mean' in X.columns:
            # Technical performance × Technical interest
            X['technical_alignment'] = X['tle_avg'] * X['interest_mean']
        
        if 'math_avg' in X.columns and 'ability_mean' in X.columns:
            # Academic performance × Ability
            X['academic_ability_alignment'] = X['math_avg'] * X['ability_mean']
    
    # Add enhanced features that better distinguish Academic vs TechPro
    X = add_enhanced_features(X, grade_cols, subject_interest_cols, ability_cols)
    
    # Scale ALL features (not just grades) for better model performance
    if is_training:
        # Use RobustScaler for better handling of outliers
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
    else:
        if scaler is not None:
            X_scaled = pd.DataFrame(
                scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X.copy()
    
    # Apply feature selection if enabled
    if USE_FEATURE_SELECTION and feature_selector is not None:
        # Convert to numpy array to avoid feature name mismatch warning
        X_selected_array = feature_selector.transform(X_scaled.values)
        selected_cols = [X_scaled.columns[i] for i in range(len(X_scaled.columns)) if feature_selector.get_support()[i]]
        X_scaled = pd.DataFrame(
            X_selected_array,
            columns=selected_cols,
            index=X_scaled.index
        )
    
    return X_scaled, grade_cols, preference_cols, scaler, list(X_scaled.columns), training_medians, feature_selector


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
    
    # Try to use model if available (prefer PKL if both exist)
    model_available = os.path.exists(MODEL_OUTPUT)
    model_pkl_available = os.path.exists(os.path.join(SCRIPT_DIR, 'strandalign_model.pkl'))
    scaler_available = os.path.exists(SCALER_OUTPUT)
    scaler_pkl_available = os.path.exists(os.path.join(SCRIPT_DIR, 'strandalign_scaler.pkl'))
    features_available = os.path.exists(FEATURES_OUTPUT)
    
    clf = None
    scaler = None
    training_features = None
    training_medians = None
    feature_selector = None
    
    # Prefer PKL files if available, otherwise use joblib
    use_pkl = model_pkl_available and scaler_pkl_available and features_available
    use_joblib = model_available and scaler_available and features_available
    
    if use_pkl or use_joblib:
        try:
            if use_pkl:
                # Load from PKL files
                model_path = os.path.join(SCRIPT_DIR, 'strandalign_model.pkl')
                scaler_path = os.path.join(SCRIPT_DIR, 'strandalign_scaler.pkl')
                with open(model_path, 'rb') as f:
                    clf = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"DEBUG: Loaded model from PKL files", file=sys.stderr)
            else:
                # Load from joblib files
                clf = joblib.load(MODEL_OUTPUT)
                scaler = joblib.load(SCALER_OUTPUT)
                print(f"DEBUG: Loaded model from joblib files", file=sys.stderr)
            
            with open(FEATURES_OUTPUT, 'r') as f:
                feature_data = json.load(f)
                training_features = feature_data.get('feature_names', [])
                training_medians = pd.Series(feature_data.get('medians', {}))
            
            # Try to load feature selector if it exists (try PKL first, then joblib)
            feature_selector_pkl_path = os.path.join(SCRIPT_DIR, 'strandalign_feature_selector.pkl')
            feature_selector_joblib_path = os.path.join(SCRIPT_DIR, 'strandalign_feature_selector.joblib')
            if os.path.exists(feature_selector_pkl_path):
                with open(feature_selector_pkl_path, 'rb') as f:
                    feature_selector = pickle.load(f)
            elif os.path.exists(feature_selector_joblib_path):
                feature_selector = joblib.load(feature_selector_joblib_path)
            
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
                
                X_row, _, _, _, _, _, _ = prepare_features(
                    row_df, 
                    drop_cols=drop_cols, 
                    is_training=False, 
                    scaler=scaler, 
                    training_medians=training_medians,
                    feature_selector=feature_selector
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
            'gender': str(gender).strip() if gender is not None else '',
            'age': int(age) if pd.notnull(age) and str(age).strip() != '' else '',
            'grade_section': grade_section,
            'recommended_track': predicted,  # Renamed from predicted_track
            'confidence': round(float(confidence), 2) if confidence is not None else 0.75
        }
        
        results.append(result_item)
    
    # Ensure excel dir exists
    os.makedirs(EXCEL_DIR, exist_ok=True)
    
    # Group results by grade_section to create separate CSV files
    df_out = pd.DataFrame(results)
    
    # If grade_section is available, group by it and create separate files
    if 'grade_section' in df_out.columns and df_out['grade_section'].notna().any():
        # Group by grade_section
        for grade_section_value, group_df in df_out.groupby('grade_section'):
            if pd.isna(grade_section_value) or str(grade_section_value).strip() == '':
                # Use default filename if grade_section is missing
                filename_safe = 'track_recommendation_Unknown.csv'
            else:
                # Clean grade_section for filename
                grade_section_clean = str(grade_section_value).strip()
                # Remove common separators and normalize
                grade_section_clean = grade_section_clean.replace(' - ', '_').replace('-', '_')
                grade_section_clean = grade_section_clean.replace(' & ', '_').replace('&', '_')
                grade_section_clean = grade_section_clean.replace(' and ', '_').replace('and', '_')
                # Replace spaces with underscores
                grade_section_clean = grade_section_clean.replace(' ', '_')
                # Remove any remaining special characters except underscores
                grade_section_clean = ''.join(c if c.isalnum() or c == '_' else '' for c in grade_section_clean)
                # Remove multiple consecutive underscores
                while '__' in grade_section_clean:
                    grade_section_clean = grade_section_clean.replace('__', '_')
                # Remove leading/trailing underscores
                grade_section_clean = grade_section_clean.strip('_')
                filename_safe = f'track_recommendation_{grade_section_clean}.csv'
            
            csv_out = os.path.join(EXCEL_DIR, filename_safe)
            # Select only required columns in order (ensure they exist)
            required_cols = ['student_id', 'gender', 'age', 'grade_section', 'recommended_track', 'confidence']
            available_cols = [col for col in required_cols if col in group_df.columns]
            output_df = group_df[available_cols].copy()
            output_df.to_csv(csv_out, index=False)
            print(f"✅ Saved CSV for {grade_section_value}: {csv_out}", file=sys.stderr)
    else:
        # If no grade_section, save single file
        filename_safe = 'track_recommendation.csv'
        csv_out = os.path.join(EXCEL_DIR, filename_safe)
        # Select only required columns in order (ensure they exist)
        required_cols = ['student_id', 'gender', 'age', 'grade_section', 'recommended_track', 'confidence']
        available_cols = [col for col in required_cols if col in df_out.columns]
        output_df = df_out[available_cols].copy()
        output_df.to_csv(csv_out, index=False)
        print(f"✅ Saved CSV: {csv_out}", file=sys.stderr)
    
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


def create_visualizations(comparison_results=None, confusion_matrices=None, feature_importance_df=None, 
                         class_dist_before=None, class_dist_after=None, cv_metrics=None):
    """
    Create comprehensive visualizations for thesis paper.
    Saves all graphs to the visualizations directory.
    """
    if not VISUALIZATION_AVAILABLE:
        print("⚠️ Visualization libraries not available. Skipping graph generation.", file=sys.stderr)
        return
    
    # Check if we should show graphs on screen
    show_graphs = globals().get('SHOW_GRAPHS', False)
    
    # Create visualizations directory
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    
    # Set style for professional-looking plots
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('ggplot')
    sns.set_palette("husl")
    
    # 1. Model Comparison Bar Chart
    if comparison_results:
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Model Comparison: Performance Metrics', fontsize=16, fontweight='bold')
            
            models = list(comparison_results.keys())
            metrics = ['accuracy_mean', 'f1_mean', 'precision_mean', 'recall_mean']
            metric_labels = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
            
            for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
                ax = axes[idx // 2, idx % 2]
                values = [comparison_results[m][metric] for m in models]
                errors = [comparison_results[m][metric.replace('_mean', '_std')] for m in models]
                
                # Highlight Random Forest
                colors = ['#2ecc71' if m == 'Random Forest' else '#3498db' for m in models]
                
                bars = ax.barh(models, values, xerr=errors, color=colors, alpha=0.8, capsize=5)
                ax.set_xlabel(f'{label} Score', fontsize=11)
                ax.set_title(f'{label} Comparison', fontsize=12, fontweight='bold')
                ax.set_xlim(0, 1.0)
                ax.grid(axis='x', alpha=0.3)
                
                # Add value labels on bars
                for i, (bar, val) in enumerate(zip(bars, values)):
                    ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'model_comparison.png'), dpi=300, bbox_inches='tight')
            if show_graphs:
                print("📊 Displaying Model Comparison chart. Close the window to continue...", file=sys.stderr)
                plt.show(block=True)  # Display and wait for user to close
            else:
                plt.close()
            print(f"✅ Saved model comparison chart: {os.path.join(VISUALIZATIONS_DIR, 'model_comparison.png')}", file=sys.stderr)
        except Exception as e:
            print(f"⚠️ Error creating model comparison chart: {str(e)}", file=sys.stderr)
    
    # 2. Confusion Matrix Heatmap
    if confusion_matrices:
        try:
            # Use the first confusion matrix or average them
            if isinstance(confusion_matrices, list):
                cm = np.mean(confusion_matrices, axis=0)
                title = 'Average Confusion Matrix (Cross-Validation)'
            else:
                cm = confusion_matrices
                title = 'Confusion Matrix'
            
            # Normalize to percentages
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            fig, ax = plt.subplots(figsize=(10, 8))
            labels = ['Academic', 'Undecided', 'TechPro']
            
            sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels,
                       cbar_kws={'label': 'Percentage (%)'}, ax=ax, vmin=0, vmax=100)
            
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
            if show_graphs:
                print("📊 Displaying Confusion Matrix. Close the window to continue...", file=sys.stderr)
                plt.show(block=True)  # Display and wait for user to close
            else:
                plt.close()
            print(f"✅ Saved confusion matrix: {os.path.join(VISUALIZATIONS_DIR, 'confusion_matrix.png')}", file=sys.stderr)
        except Exception as e:
            print(f"⚠️ Error creating confusion matrix: {str(e)}", file=sys.stderr)
    
    # 3. Feature Importance Chart
    if feature_importance_df is not None and len(feature_importance_df) > 0:
        try:
            # Get top 15 features
            top_features = feature_importance_df.head(15)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
            
            bars = ax.barh(range(len(top_features)), top_features['Importance (%)'].values, color=colors)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['Feature'].values, fontsize=9)
            ax.set_xlabel('Importance (%)', fontsize=12)
            ax.set_title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, top_features['Importance (%)'].values)):
                ax.text(val + 0.2, i, f'{val:.2f}%', va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
            if show_graphs:
                print("📊 Displaying Feature Importance chart. Close the window to continue...", file=sys.stderr)
                plt.show(block=True)  # Display and wait for user to close
            else:
                plt.close()
            print(f"✅ Saved feature importance chart: {os.path.join(VISUALIZATIONS_DIR, 'feature_importance.png')}", file=sys.stderr)
        except Exception as e:
            print(f"⚠️ Error creating feature importance chart: {str(e)}", file=sys.stderr)
    
    # 4. Class Distribution Before/After Augmentation
    if class_dist_before and class_dist_after:
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Before augmentation
            classes = list(class_dist_before.keys())
            counts_before = [class_dist_before[c] for c in classes]
            colors_map = {'Academic': '#3498db', 'TechPro': '#e74c3c', 'Undecided': '#f39c12'}
            bar_colors = [colors_map.get(c, '#95a5a6') for c in classes]
            
            ax1.bar(classes, counts_before, color=bar_colors, alpha=0.8)
            ax1.set_title('Class Distribution (Before Augmentation)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Number of Samples', fontsize=11)
            ax1.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, (cls, count) in enumerate(zip(classes, counts_before)):
                ax1.text(i, count + 5, str(count), ha='center', fontsize=10, fontweight='bold')
            
            # After augmentation
            counts_after = [class_dist_after[c] for c in classes]
            ax2.bar(classes, counts_after, color=bar_colors, alpha=0.8)
            ax2.set_title('Class Distribution (After Augmentation)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Number of Samples', fontsize=11)
            ax2.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, (cls, count) in enumerate(zip(classes, counts_after)):
                ax2.text(i, count + 5, str(count), ha='center', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'class_distribution.png'), dpi=300, bbox_inches='tight')
            if show_graphs:
                print("📊 Displaying Class Distribution chart. Close the window to continue...", file=sys.stderr)
                plt.show(block=True)  # Display and wait for user to close
            else:
                plt.close()
            print(f"✅ Saved class distribution chart: {os.path.join(VISUALIZATIONS_DIR, 'class_distribution.png')}", file=sys.stderr)
        except Exception as e:
            print(f"⚠️ Error creating class distribution chart: {str(e)}", file=sys.stderr)
    
    # 5. Cross-Validation Metrics
    if cv_metrics:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
            means = [cv_metrics.get('mean_accuracy', 0), cv_metrics.get('mean_f1', 0),
                    cv_metrics.get('mean_precision', 0), cv_metrics.get('mean_recall', 0)]
            stds = [cv_metrics.get('std_accuracy', 0), cv_metrics.get('std_f1', 0),
                   cv_metrics.get('std_precision', 0), cv_metrics.get('std_recall', 0)]
            
            x_pos = np.arange(len(metrics))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=10, color='#2ecc71', alpha=0.8, edgecolor='black')
            
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title('Cross-Validation Performance Metrics', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1.0)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                ax.text(i, mean + std + 0.02, f'{mean:.3f}±{std:.3f}', 
                       ha='center', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'cv_metrics.png'), dpi=300, bbox_inches='tight')
            if show_graphs:
                print("📊 Displaying Cross-Validation Metrics chart. Close the window to continue...", file=sys.stderr)
                plt.show(block=True)  # Display and wait for user to close
            else:
                plt.close()
            print(f"✅ Saved CV metrics chart: {os.path.join(VISUALIZATIONS_DIR, 'cv_metrics.png')}", file=sys.stderr)
        except Exception as e:
            print(f"⚠️ Error creating CV metrics chart: {str(e)}", file=sys.stderr)
    
    print(f"\n📊 All visualizations saved to: {VISUALIZATIONS_DIR}", file=sys.stderr)


def compare_models(X, y, class_weights_dict=None, cv_folds=3):
    """
    Compare multiple classification models to demonstrate Random Forest superiority.
    Returns a dictionary with model comparison results.
    """
    print("\n" + "="*80, file=sys.stderr)
    print("🔬 MODEL COMPARISON: Evaluating Multiple Classifiers", file=sys.stderr)
    print("="*80, file=sys.stderr)
    
    # Define model configurations (will create fresh instances for each fold)
    model_configs = {
        'Dummy Classifier (Baseline)': {
            'class': DummyClassifier,
            'params': {'strategy': 'stratified', 'random_state': RANDOM_STATE}
        },
        'Logistic Regression': {
            'class': LogisticRegression,
            'params': {
                'max_iter': 1000,
                'random_state': RANDOM_STATE,
                'class_weight': class_weights_dict if class_weights_dict else 'balanced',
                'solver': 'lbfgs'
            }
        },
        'Support Vector Machine (SVM)': {
            'class': SVC,
            'params': {
                'kernel': 'rbf',
                'probability': True,
                'random_state': RANDOM_STATE,
                'class_weight': class_weights_dict if class_weights_dict else 'balanced'
            }
        },
        'Gradient Boosting': {
            'class': GradientBoostingClassifier,
            'params': {
                'n_estimators': 100,
                'max_depth': 5,
                'random_state': RANDOM_STATE,
                'learning_rate': 0.1
            }
        },
        'Random Forest': {
            'class': RandomForestClassifier,
            'params': {
                'n_estimators': CV_N_ESTIMATORS,
                'max_depth': 25,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': RANDOM_STATE,
                'class_weight': class_weights_dict if class_weights_dict else 'balanced',
                'n_jobs': -1
            }
        }
    }
    
    # Cross-validation setup
    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    
    # Store results for each model
    results = {}
    
    for model_name, config in model_configs.items():
        print(f"\n📊 Evaluating {model_name}...", file=sys.stderr)
        
        fold_accuracies = []
        fold_f1s = []
        fold_precisions = []
        fold_recalls = []
        
        for fold_num, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Balance training data with SMOTE (except for Dummy Classifier)
            if model_name != 'Dummy Classifier (Baseline)':
                min_class_size = min(Counter(y_train).values())
                if min_class_size >= 2:
                    smote = SMOTE(random_state=RANDOM_STATE)
                    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
                else:
                    X_train_bal, y_train_bal = X_train, y_train
            else:
                X_train_bal, y_train_bal = X_train, y_train
            
            # Train and predict
            try:
                # Create a fresh model instance for this fold
                model = config['class'](**config['params'])
                model.fit(X_train_bal, y_train_bal)
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                
                fold_accuracies.append(acc)
                fold_f1s.append(f1)
                fold_precisions.append(prec)
                fold_recalls.append(rec)
            except Exception as e:
                print(f"⚠️ Error in fold {fold_num} for {model_name}: {str(e)}", file=sys.stderr)
                fold_accuracies.append(0.0)
                fold_f1s.append(0.0)
                fold_precisions.append(0.0)
                fold_recalls.append(0.0)
        
        # Calculate mean and std
        results[model_name] = {
            'accuracy_mean': np.mean(fold_accuracies),
            'accuracy_std': np.std(fold_accuracies),
            'f1_mean': np.mean(fold_f1s),
            'f1_std': np.std(fold_f1s),
            'precision_mean': np.mean(fold_precisions),
            'precision_std': np.std(fold_precisions),
            'recall_mean': np.mean(fold_recalls),
            'recall_std': np.std(fold_recalls)
        }
        
        print(f"  Accuracy: {results[model_name]['accuracy_mean']:.4f} ± {results[model_name]['accuracy_std']:.4f}", file=sys.stderr)
        print(f"  F1-Score: {results[model_name]['f1_mean']:.4f} ± {results[model_name]['f1_std']:.4f}", file=sys.stderr)
    
    # Print comparison table
    print("\n" + "="*80, file=sys.stderr)
    print("📊 MODEL COMPARISON RESULTS", file=sys.stderr)
    print("="*80, file=sys.stderr)
    print(f"{'Model':<35} {'Accuracy':<15} {'F1-Score':<15} {'Precision':<15} {'Recall':<15}", file=sys.stderr)
    print("-"*80, file=sys.stderr)
    
    # Sort by F1 score (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1_mean'], reverse=True)
    
    for model_name, metrics in sorted_results:
        acc_str = f"{metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}"
        f1_str = f"{metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}"
        prec_str = f"{metrics['precision_mean']:.4f} ± {metrics['precision_std']:.4f}"
        rec_str = f"{metrics['recall_mean']:.4f} ± {metrics['recall_std']:.4f}"
        
        # Highlight Random Forest
        marker = "🏆" if model_name == 'Random Forest' else "  "
        print(f"{marker} {model_name:<33} {acc_str:<15} {f1_str:<15} {prec_str:<15} {rec_str:<15}", file=sys.stderr)
    
    print("="*80, file=sys.stderr)
    
    # Verify Random Forest is best
    rf_f1 = results['Random Forest']['f1_mean']
    best_model = max(sorted_results, key=lambda x: x[1]['f1_mean'])
    
    if best_model[0] == 'Random Forest':
        print(f"\n✅ Random Forest is the BEST performing model!", file=sys.stderr)
        print(f"   F1-Score: {rf_f1:.4f} (highest among all models)", file=sys.stderr)
    else:
        print(f"\n⚠️ Note: {best_model[0]} achieved highest F1-Score ({best_model[1]['f1_mean']:.4f})", file=sys.stderr)
        print(f"   Random Forest F1-Score: {rf_f1:.4f}", file=sys.stderr)
    
    print("="*80 + "\n", file=sys.stderr)
    
    return results


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
        if a >= 3.5 and t < 3.5:
            return "Academic"
        elif t >= 3.5 and a < 3.5:
            return "TechPro"
        elif t >= 3.5 and a >= 3.5:
            return "Undecided"
        elif abs(a - t) <= 0.5 and (a >= 3 or t >= 3):
            # Only truly balanced cases become Undecided
            return "Undecided"
        elif a >= 3 and t >= 3:
            # Both high but not balanced - pick the higher one
            return "Academic" if a > t else "TechPro"
        else:
            # Low scores - use the higher one or default to Undecided
            if max(a, t) >= 2.5:
                return "Academic" if a > t else "TechPro"
            return "Undecided"
    
    df['derived_group_label'] = df.apply(derive_group_label, axis=1)
    print("🎯 Derived group label distribution (before augmentation):", file=sys.stderr)
    class_dist_before = df['derived_group_label'].value_counts().to_dict()
    print(df['derived_group_label'].value_counts(), file=sys.stderr)
    print("", file=sys.stderr)
    
    # Data quality improvements
    if ENABLE_DATA_QUALITY_CHECKS:
        print("🔍 Performing data quality checks...", file=sys.stderr)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df = improve_data_quality(df, numeric_cols)
    
    # Augment classes to balance dataset
    if BALANCE_ALL_CLASSES_EQUALLY:
        print(f"\n⚖️ Balancing all classes to equal size...", file=sys.stderr)
        class_counts = df['derived_group_label'].value_counts().to_dict()
        max_class_size = max(class_counts.values())
        print(f"   Target size (largest class): {max_class_size}", file=sys.stderr)
        print(f"   Current distribution: {class_counts}", file=sys.stderr)
        
        # Augment each class to match the largest class size
        for class_name in class_counts.keys():
            current_size = class_counts[class_name]
            if current_size < max_class_size:
                # Calculate augmentation factor needed to reach max_class_size
                augmentation_factor = max_class_size / current_size
                class_mask = df['derived_group_label'] == class_name
                print(f"\n📈 Augmenting {class_name} class: {current_size} → {max_class_size} (factor: {augmentation_factor:.2f})", file=sys.stderr)
                df = augment_class_data(df, class_mask, class_name, augmentation_factor, enabled=True)
                # Re-derive labels for augmented data
                df['derived_group_label'] = df.apply(derive_group_label, axis=1)
            else:
                print(f"   {class_name} class already at target size ({current_size})", file=sys.stderr)
    else:
        # Original augmentation logic (using fixed factors)
        # Augment Academic class data
        academic_mask = df['derived_group_label'] == 'Academic'
        if AUGMENT_ACADEMIC_DATA and academic_mask.sum() > 0:
            print(f"\n📈 Augmenting Academic class data...", file=sys.stderr)
            df = augment_class_data(df, academic_mask, "Academic", ACADEMIC_AUGMENTATION_FACTOR, AUGMENT_ACADEMIC_DATA)
            # Re-derive labels for augmented data
            df['derived_group_label'] = df.apply(derive_group_label, axis=1)
        
        # Augment TechPro class data
        techpro_mask = df['derived_group_label'] == 'TechPro'
        if AUGMENT_TECHPRO_DATA and techpro_mask.sum() > 0:
            print(f"\n📈 Augmenting TechPro class data...", file=sys.stderr)
            df = augment_class_data(df, techpro_mask, "TechPro", TECHPRO_AUGMENTATION_FACTOR, AUGMENT_TECHPRO_DATA)
            # Re-derive labels for augmented data
            df['derived_group_label'] = df.apply(derive_group_label, axis=1)
        
        # Augment Undecided class data if enabled
        if AUGMENT_UNDECIDED_DATA:
            undecided_mask = df['derived_group_label'] == 'Undecided'
            if undecided_mask.sum() > 0:
                print(f"\n📈 Augmenting Undecided class data...", file=sys.stderr)
                # Use average of other factors for Undecided
                undecided_factor = (ACADEMIC_AUGMENTATION_FACTOR + TECHPRO_AUGMENTATION_FACTOR) / 2
                df = augment_class_data(df, undecided_mask, "Undecided", undecided_factor, AUGMENT_UNDECIDED_DATA)
                # Re-derive labels for augmented data
                df['derived_group_label'] = df.apply(derive_group_label, axis=1)
    
    # Show final distribution after all augmentations
    class_dist_after = df['derived_group_label'].value_counts().to_dict()
    if BALANCE_ALL_CLASSES_EQUALLY or (AUGMENT_ACADEMIC_DATA or AUGMENT_TECHPRO_DATA or AUGMENT_UNDECIDED_DATA):
        print("\n🎯 Derived group label distribution (after augmentation):", file=sys.stderr)
        print(df['derived_group_label'].value_counts(), file=sys.stderr)
        print("", file=sys.stderr)
    
    # Calculate class weights for better class performance
    class_weights_dict = None
    if USE_CLASS_WEIGHT_ADJUSTMENT:
        print("\n⚖️ Calculating class weights...", file=sys.stderr)
        from sklearn.utils.class_weight import compute_class_weight
        classes = df['derived_group_label'].unique()
        class_weights = compute_class_weight('balanced', classes=classes, y=df['derived_group_label'])
        class_weights_dict = dict(zip(classes, class_weights))
        
        # Boost Academic class weight further if needed (it was underperforming)
        if 'Academic' in class_weights_dict:
            # Increase Academic weight by 20% to improve its performance
            class_weights_dict['Academic'] *= 1.2
        
        # Boost TechPro class weight to improve its performance
        if BOOST_TECHPRO_WEIGHT and 'TechPro' in class_weights_dict:
            class_weights_dict['TechPro'] *= TECHPRO_WEIGHT_BOOST
            print(f"✅ Boosted TechPro class weight by {((TECHPRO_WEIGHT_BOOST - 1) * 100):.0f}%", file=sys.stderr)
        
        print(f"Class weights: {class_weights_dict}", file=sys.stderr)
    
    # Prepare features
    drop_cols = [academic_col, techpro_col, 'derived_group_label']
    X, grade_cols, preference_cols, scaler, feature_names, training_medians, _ = prepare_features(
        df, drop_cols=drop_cols, is_training=True
    )
    y = df['derived_group_label']
    
    print(f"✅ Prepared {len(feature_names)} features for training.", file=sys.stderr)
    if USE_ADVANCED_FEATURES:
        print(f"   (Includes advanced features: aggregations, ratios, interactions, statistics)", file=sys.stderr)
    if USE_ENHANCED_FEATURES:
        print(f"   (Includes enhanced features: Academic vs TechPro distinguishing features)", file=sys.stderr)
    print(f"Sample columns: {feature_names[:10]}\n", file=sys.stderr)
    
    # Balance full dataset for hyperparameter search and feature selection
    print("⚖️ Balancing dataset for hyperparameter search...", file=sys.stderr)
    min_class_size = min(Counter(y).values())
    if min_class_size >= 2:
        smote_search = SMOTE(random_state=RANDOM_STATE)
        X_bal_search, y_bal_search = smote_search.fit_resample(X, y)
        print(f"✅ Balanced dataset: {dict(Counter(y_bal_search))}", file=sys.stderr)
    else:
        X_bal_search, y_bal_search = X, y
        print("⚠️ Not enough samples for SMOTE. Proceeding without balancing.", file=sys.stderr)
    
    # Improved hyperparameter tuning (ONCE, outside CV loop for efficiency)
    # Check if we can skip hyperparameter search by loading previous best params
    best_params_global = None
    best_params_file = os.path.join(SCRIPT_DIR, 'strandalign_best_params.json')
    
    if SKIP_HYPERPARAM_SEARCH_IF_MODEL_EXISTS and os.path.exists(best_params_file) and os.path.exists(MODEL_OUTPUT):
        try:
            print(f"\n⚡ Skipping hyperparameter search (using previous best params)...", file=sys.stderr)
            with open(best_params_file, 'r') as f:
                saved_params = json.load(f)
                # Convert class_weight dict back if it exists
                if 'class_weight' in saved_params and isinstance(saved_params['class_weight'], dict):
                    # Keep as dict (will be handled by RandomForestClassifier)
                    pass
                best_params_global = saved_params
            print(f"✅ Loaded previous best params: {best_params_global}", file=sys.stderr)
        except Exception as e:
            print(f"⚠️ Failed to load previous params: {str(e)}. Proceeding with hyperparameter search...", file=sys.stderr)
            best_params_global = None
    
    # Adjust settings based on QUICK_TRAINING_MODE
    # Prepare class_weight options
    class_weight_options = ['balanced', 'balanced_subsample', None]
    if USE_CLASS_WEIGHT_ADJUSTMENT and class_weights_dict is not None:
        # Add custom class weights to search space
        class_weight_options.append(class_weights_dict)
    
    if QUICK_TRAINING_MODE:
        search_iterations = 20  # Reduced from 100
        cv_folds_search = 2  # Reduced from 3
        param_grid = {
            'n_estimators': [300, 500, 700],  # Reduced options
            'max_depth': [25, 35, None],  # Reduced options
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True],
            'class_weight': class_weight_options[:3] if len(class_weight_options) > 3 else class_weight_options  # Limit for quick mode
        }
        print(f"\n🔍 Quick mode: Hyperparameter tuning ({search_iterations} iterations)...", file=sys.stderr)
    else:
        search_iterations = HYPERPARAM_SEARCH_ITERATIONS
        cv_folds_search = 2  # Reduced from 3 for faster search (still valid)
        # Optimized grid: fewer n_estimators options, but still good coverage
        param_grid = {
            'n_estimators': [300, 500, 700, 1000],  # Reduced from 5 to 4 options
            'max_depth': [20, 25, 35, 50, None],  # Slightly reduced
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'class_weight': class_weight_options
        }
        print(f"\n🔍 Hyperparameter tuning ({search_iterations} iterations, {cv_folds_search} CV folds - optimized for speed)...", file=sys.stderr)
    
    # Only run hyperparameter search if we don't have previous best params
    if best_params_global is None:
        clf_base = RandomForestClassifier(random_state=RANDOM_STATE)
        search = RandomizedSearchCV(
            clf_base, param_grid, 
            n_iter=search_iterations, 
            scoring='f1_weighted', 
            cv=cv_folds_search,
            random_state=RANDOM_STATE, 
            n_jobs=-1, 
            verbose=1
        )
        search.fit(X_bal_search, y_bal_search)
        best_params_global = search.best_params_
        print(f"✅ Best hyperparameters found: {best_params_global}", file=sys.stderr)
        
        # Save best params for future use
        try:
            # Convert numpy types to native Python types for JSON serialization
            params_to_save = {}
            for key, value in best_params_global.items():
                if isinstance(value, (np.integer, np.floating)):
                    params_to_save[key] = value.item()
                elif isinstance(value, dict):
                    # Handle class_weight dict
                    params_to_save[key] = {k: (v.item() if isinstance(v, (np.integer, np.floating)) else v) 
                                          for k, v in value.items()}
                else:
                    params_to_save[key] = value
            with open(best_params_file, 'w') as f:
                json.dump(params_to_save, f, indent=2)
            print(f"💾 Saved best params to {best_params_file} for faster future training", file=sys.stderr)
        except Exception as e:
            print(f"⚠️ Could not save best params: {str(e)}", file=sys.stderr)
    else:
        skipped_fits = search_iterations * cv_folds_search if 'search_iterations' in locals() else 0
        print(f"✅ Using loaded best params (skipped ~{skipped_fits} hyperparameter search fits!)", file=sys.stderr)
    
    # Feature selection based on importance or k-best
    feature_selector = None
    if USE_FEATURE_SELECTION:
        print(f"\n🎯 Performing feature selection (method: {FEATURE_SELECTION_METHOD})...", file=sys.stderr)
        if FEATURE_SELECTION_METHOD == 'importance':
            # Train a quick model to get feature importances
            selector_params = dict(best_params_global) if best_params_global else {}
            selector_params['n_estimators'] = 100  # Use fewer for speed
            selector_params['random_state'] = RANDOM_STATE
            selector_rf = RandomForestClassifier(**selector_params)
            # Fit on DataFrame to preserve feature names
            selector_rf.fit(X, y)
            feature_selector = SelectFromModel(selector_rf, threshold=FEATURE_SELECTION_THRESHOLD, prefit=True)
            # Convert to numpy array to avoid feature name mismatch warning
            X_selected = feature_selector.transform(X.values)
            selected_features = [feature_names[i] for i in range(len(feature_names)) if feature_selector.get_support()[i]]
            print(f"✅ Selected {len(selected_features)} features (from {len(feature_names)}, threshold={FEATURE_SELECTION_THRESHOLD})", file=sys.stderr)
        elif FEATURE_SELECTION_METHOD == 'kbest':
            # Use k-best features based on F-statistic
            k_best = min(K_BEST_FEATURES, len(feature_names))
            feature_selector = SelectKBest(f_classif, k=k_best)
            feature_selector.fit(X, y)
            X_selected = feature_selector.transform(X)
            selected_features = [feature_names[i] for i in range(len(feature_names)) if feature_selector.get_support()[i]]
            print(f"✅ Selected {len(selected_features)} features (top {k_best} based on F-statistic)", file=sys.stderr)
        
        X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        feature_names = selected_features

    # Model comparison: Compare multiple models to show Random Forest is best
    print("\n" + "="*80, file=sys.stderr)
    print("🔬 MODEL COMPARISON PHASE", file=sys.stderr)
    print("="*80, file=sys.stderr)
    comparison_results = compare_models(X, y, class_weights_dict=class_weights_dict, cv_folds=3)
    
    # Add comparison results to summary
    comparison_summary = {}
    for model_name, metrics in comparison_results.items():
        comparison_summary[model_name] = {
            'accuracy': round(metrics['accuracy_mean'], 4),
            'f1_score': round(metrics['f1_mean'], 4),
            'precision': round(metrics['precision_mean'], 4),
            'recall': round(metrics['recall_mean'], 4)
        }

    # Cross-validation with enhanced metrics
    # Adjust folds based on QUICK_TRAINING_MODE
    cv_folds = 3 if QUICK_TRAINING_MODE else N_SPLITS
    print(f"\n📊 Starting {cv_folds}-fold cross-validation...", file=sys.stderr)
    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_accuracies, fold_f1s, fold_precisions, fold_recalls = [], [], [], []
    confusion_matrices = []  # Store confusion matrices for visualization

    for fold_num, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
        print(80 * "=", file=sys.stderr)
        print(f"Fold {fold_num}/{cv_folds}", file=sys.stderr)
        print(80 * "=", file=sys.stderr)

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        classcounts = Counter(y_train)
        print(f"Original class distribution: {dict(classcounts)}", file=sys.stderr)

        # Apply SMOTE for balancing if applicable
        if min(classcounts.values()) >= 2:
            smote = SMOTE(random_state=RANDOM_STATE)
            X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
            print(f"Balanced class distribution: {dict(Counter(y_train_bal))}", file=sys.stderr)
        else:
            X_train_bal, y_train_bal = X_train, y_train
            print("Not enough minority class samples for SMOTE. Proceeding without balancing.", file=sys.stderr)

        # Use best params from hyperparameter search, but with fewer trees for CV speed
        cv_params = best_params_global.copy() if best_params_global else {}
        # Override n_estimators for faster CV (final model will use full n_estimators)
        cv_params['n_estimators'] = CV_N_ESTIMATORS
        cv_params['random_state'] = RANDOM_STATE
        best_clf = RandomForestClassifier(**cv_params)
        best_clf.fit(X_train_bal, y_train_bal)

        y_pred = best_clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        fold_accuracies.append(acc)
        fold_f1s.append(f1)
        fold_precisions.append(prec)
        fold_recalls.append(rec)
        confusion_matrices.append(cm)  # Store for visualization

        print(f"Accuracy: {acc:.4f}, Weighted F1: {f1:.4f}", file=sys.stderr)
        print(f"Precision: {prec:.4f}, Recall: {rec:.4f}", file=sys.stderr)
        print("Confusion Matrix:", file=sys.stderr)
        print(cm, file=sys.stderr)

    # Calculate mean and standard deviation of metrics after cross-validation
    meanacc = np.mean(fold_accuracies)
    stdacc = np.std(fold_accuracies)
    meanf1 = np.mean(fold_f1s)
    stdf1 = np.std(fold_f1s)
    meanprec = np.mean(fold_precisions)
    stdprec = np.std(fold_precisions)
    meanrec = np.mean(fold_recalls)
    stdrec = np.std(fold_recalls)

    # Now use these variables for reporting or summary logs
    print(f"\n📊 Cross-Validation Summary ({cv_folds} folds):", file=sys.stderr)
    print(f"Mean Accuracy: {meanacc:.4f} ± {stdacc:.4f}", file=sys.stderr)
    print(f"Mean Weighted F1: {meanf1:.4f} ± {stdf1:.4f}", file=sys.stderr)
    print(f"Mean Precision: {meanprec:.4f} ± {stdprec:.4f}", file=sys.stderr)
    print(f"Mean Recall: {meanrec:.4f} ± {stdrec:.4f}", file=sys.stderr)


    
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
    best_params = best_params_global.copy() if best_params_global is not None else {
        'n_estimators': 400,
        'max_depth': 40,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'bootstrap': True,
        'class_weight': 'balanced_subsample'
    }
    
    # Override class_weight with custom weights if enabled and not already optimal
    if USE_CLASS_WEIGHT_ADJUSTMENT and class_weights_dict is not None:
        # If best params have None or 'balanced', try custom weights
        if best_params.get('class_weight') in [None, 'balanced']:
            best_params['class_weight'] = class_weights_dict
            print(f"✅ Using custom class weights for final model: {class_weights_dict}", file=sys.stderr)
    
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
    
    # Generate visualizations for thesis paper
    print("\n📊 Generating visualizations for thesis paper...", file=sys.stderr)
    cv_metrics_dict = {
        'mean_accuracy': meanacc,
        'std_accuracy': stdacc,
        'mean_f1': meanf1,
        'std_f1': stdf1,
        'mean_precision': meanprec,
        'std_precision': stdprec,
        'mean_recall': meanrec,
        'std_recall': stdrec
    }
    create_visualizations(
        comparison_results=comparison_results if 'comparison_results' in locals() else None,
        confusion_matrices=confusion_matrices if 'confusion_matrices' in locals() else None,
        feature_importance_df=feat_imp,
        class_dist_before=class_dist_before if 'class_dist_before' in locals() else None,
        class_dist_after=class_dist_after if 'class_dist_after' in locals() else None,
        cv_metrics=cv_metrics_dict
    )
    
    # Save model, scaler, feature selector, and feature names
    joblib.dump(calibrated_rf, MODEL_OUTPUT)
    joblib.dump(scaler, SCALER_OUTPUT)
    if feature_selector is not None:
        feature_selector_path = os.path.join(SCRIPT_DIR, 'strandalign_feature_selector.joblib')
        joblib.dump(feature_selector, feature_selector_path)
        print(f"✅ Feature selector saved: {feature_selector_path}", file=sys.stderr)
    
    # Save as PKL files if enabled
    if SAVE_AS_PKL:
        model_pkl_path = os.path.join(SCRIPT_DIR, 'strandalign_model.pkl')
        scaler_pkl_path = os.path.join(SCRIPT_DIR, 'strandalign_scaler.pkl')
        
        with open(model_pkl_path, 'wb') as f:
            pickle.dump(calibrated_rf, f)
        print(f"✅ Model saved as PKL: {model_pkl_path}", file=sys.stderr)
        
        with open(scaler_pkl_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✅ Scaler saved as PKL: {scaler_pkl_path}", file=sys.stderr)
        
        if feature_selector is not None:
            feature_selector_pkl_path = os.path.join(SCRIPT_DIR, 'strandalign_feature_selector.pkl')
            with open(feature_selector_pkl_path, 'wb') as f:
                pickle.dump(feature_selector, f)
            print(f"✅ Feature selector saved as PKL: {feature_selector_pkl_path}", file=sys.stderr)
    
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
            
            # Prepare features using the same pipeline as training
            drop_cols_test = []
            X_test_prepared, _, _, _, _, _, _ = prepare_features(
                df_test,
                drop_cols=drop_cols_test,
                is_training=False,
                scaler=scaler,
                training_medians=training_medians,
                feature_selector=feature_selector
            )
            
            # Align with training features
            X_test_prepared = X_test_prepared.reindex(columns=feature_names, fill_value=0)
            X_test_prepared = X_test_prepared.fillna(0)
            
            y_pred_test = calibrated_rf.predict(X_test_prepared)
            y_pred_prob = calibrated_rf.predict_proba(X_test_prepared)
            
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
        'mean_accuracy': round(meanacc, 4),
        'std_accuracy': round(stdacc, 4),
        'mean_f1_weighted': round(meanf1, 4),
        'std_f1_weighted': round(stdf1, 4),
        'mean_precision': round(meanprec, 4),
        'std_precision': round(stdprec, 4),
        'mean_recall': round(meanrec, 4),
        'std_recall': round(stdrec, 4),
        'num_features': len(feature_names),
        'best_params': best_params,
        'model_comparison': comparison_summary if 'comparison_summary' in locals() else {}
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
