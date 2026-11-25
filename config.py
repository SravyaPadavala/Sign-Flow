"""
SignFlow++ Configuration Module
Defines dataset settings, vocabulary mappings for Indian Sign Language (ISL),
hyperparameters for both the Base Paper model (SignFlow) and the Improved Model (SignFlow++).
"""

import os

# Root directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------
# Vocabulary Settings (eRaktkosh & UMANG ISL Domain Corpus)
# -------------------------------------------------------------
# Standard blank symbol is index 0 for CTC loss
BLANK_TOKEN = "<BLANK>"

ISL_GLOSS_VOCAB = [
    BLANK_TOKEN,
    # Common core ISL words in eRaktkosh & medical emergency domain
    "BLOOD", "DONATE", "DONATION", "DONOR", "REGISTER", "REGISTRATION",
    "HOSPITAL", "CAMP", "ONLINE", "BOOK", "APPOINTMENT", "CARD", "CERTIFICATE",
    "HEMOGLOBIN", "TEST", "WEIGHT", "AGE", "ELIGIBLE", "ELIGIBILITY", "CRITERIA",
    "RARE", "GROUP", "A_POSITIVE", "B_POSITIVE", "O_POSITIVE", "AB_POSITIVE",
    "A_NEGATIVE", "B_NEGATIVE", "O_NEGATIVE", "AB_NEGATIVE", "URGENT", "EMERGENCY",
    "PLASMA", "PLATELETS", "WB", "RBC", "AVAILABLE", "STOCK", "STATUS",
    "LOCATION", "NEAR", "CITY", "STATE", "PINCODE", "CONTACT", "NUMBER",
    "TIME", "DATE", "TODAY", "TOMORROW", "HOURS", "HELP", "REQUEST",
    "SEARCH", "FIND", "SERVICE", "PROCESS", "HOW", "WHERE", "WHEN",
    "WHY", "YES", "NO", "MALE", "FEMALE", "GENDER", "INTERVAL", "MONTHS",
    "DAYS", "AFTER", "BEFORE", "FEVER", "MEDICINE", "DIABETES", "PRESSURE",
    "TATTOO", "TRAVEL", "ALCOHOL", "SMOKING", "PREGNANT", "SAFE", "FREE",
    "GOVERNMENT", "APPROVAL", "VERIFY", "OTP", "LOGIN", "SUBMIT", "CANCEL",
    "DOWNLOAD", "PRINT", "SHARE", "UPDATE", "PROFILE", "NAME", "AADHAAR",
    "REPORT", "FEEDBACK", "THANK_YOU", "WELCOME", "HELLO", "BYE"
]

GLOSS_TO_IDX = {gloss: idx for idx, gloss in enumerate(ISL_GLOSS_VOCAB)}
IDX_TO_GLOSS = {idx: gloss for idx, gloss in enumerate(ISL_GLOSS_VOCAB)}
VOCAB_SIZE = len(ISL_GLOSS_VOCAB)

# -------------------------------------------------------------
# Preprocessing Specifications (Base Paper vs Proposed)
# -------------------------------------------------------------
FRAME_HEIGHT = 112
FRAME_WIDTH = 112
CHANNELS = 3

# Base paper downsampling settings
TARGET_FPS = 6
TRAIN_FPS_RANGE = (5, 8)
MAX_SEQUENCE_LENGTH = 250  # Padded frame length

# Landmark Feature Dimensions (Upper body pose + hands = 443 features)
# MediaPipe Pose upper-body: 22 joint coordinates (x, y, z) + bone lengths + relative positions
# Left Hand: 21 keypoints (x, y, z) + bone lengths + relative positions
# Right Hand: 21 keypoints (x, y, z) + bone lengths + relative positions
# Inter-hand & Joint Euclidean distances
LANDMARK_FEAT_DIM = 443

# -------------------------------------------------------------
# Model Hyperparameters
# -------------------------------------------------------------
# Common
D_MODEL = 512
NUM_HEADS = 8
DROPOUT = 0.1

# Base Paper (SignFlow) Architecture:
BASE_CONFIG = {
    "name": "SignFlow_Base",
    "cnn_backbone": "mc3_18",  # Mixed Convolution 3D ResNet-18
    "transformer_layers": 8,
    "d_model": D_MODEL,
    "num_heads": NUM_HEADS,
    "dropout": DROPOUT,
    "use_positional_encoding": False,  # Paper tested both; False gave higher detection rate
    "learning_rate_cnn": 1e-4,
    "learning_rate_transformer": 1e-3,
    "batch_size": 4,
    "beam_size": 5,
    "length_penalty_alpha": 1.0,
}

# Proposed High-Accuracy Model (SignFlow++):
PLUSPLUS_CONFIG = {
    "name": "SignFlow_PlusPlus",
    "rgb_backbone": "MultiScale3DCNN",  # Multi-Scale Temporal Convolutions
    "pose_backbone": "SpatioTemporalGCN",  # 3D Skeletal Landmark Temporal Encoder
    "d_model": D_MODEL,
    "num_heads": NUM_HEADS,
    "conformer_layers": 6,  # 6 Conformer blocks (Self-Attention + Depthwise Conv + Macaron FFN)
    "temporal_dilations": [1, 2, 4],  # Multi-scale temporal receptive fields for fast signing
    "use_relative_pos_bias": True,  # Relative positional encoding (preserves sequence order & keywords)
    "use_cross_modal_attention": True,  # Dual-stream RGB <-> Pose cross-attention
    "use_vac": True,  # Visual Alignment Constraint loss
    "loss_weights": {
        "fused_ctc": 1.0,
        "aux_visual_ctc": 0.3,
        "aux_pose_ctc": 0.3,
        "vac_alignment": 0.15,
    },
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "batch_size": 4,
    "beam_size": 8,
    "length_penalty_alpha": 0.8,
    "lm_weight_beta": 0.3,  # N-gram domain language model rescoring weight
}
