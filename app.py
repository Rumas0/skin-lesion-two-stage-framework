import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import streamlit as st
import torchvision.transforms as transforms

# ==============================================================================
# 1. PAGE CONFIGURATION AND CLINICAL THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="Two-Stage Medical DL Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional desaturated slate-blue color palette for an advanced medical research application
st.markdown("""
    <style>
    .reportview-container .main .block-container { padding-top: 1.5rem; }
    body { color: #2d3748; background-color: #f7fafc; }
    .metric-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }
    .custom-header {
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 25px;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 50px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
    }
    .badge-success { background-color: #def7ec; color: #03543f; }
    .badge-warning { background-color: #fde8e8; color: #9b1c1c; }
    .badge-info { background-color: #e1effe; color: #1e429f; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CORE SYSTEM ARCHITECTURE (Strict Alignment with Thesis Notebooks)
# ==============================================================================
CLASS_NAMES = ['BKL', 'MEL', 'NV']
CLASS_LABELS = {
    'BKL': 'Benign Keratosis-like Lesions (Seborrheic Keratoses / Solar Lentigines)',
    'MEL': 'Malignant Melanoma (High-Risk Skin Cancer Category)',
    'NV': 'Melanocytic Nevi (Common Benign Moles)'
}

class SimpleEncoder(nn.Module):
    """
    Structural backbone exactly mapping to your thesis repository.
    Handles feature maps through 3 sequential convolutional layer matrices.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), # 112x112
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), # 56x56
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2), # 28x28
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten() # Extracted embedding feature space dim = 128
        )
        
    def forward(self, x):
        return self.features(x)

class SSLClassifier(nn.Module):
    """
    Complete target framework mapping features from the learned encoder latent space
    to the downstream clinical target diagnostic probability scores.
    """
    def __init__(self, encoder, num_classes=3):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(self.encoder(x))

# ==============================================================================
# 3. ROBUST PARAMETER STATE ALIGNMENT LOADER (Resolves State Dict Mismatches)
# ==============================================================================
@st.cache_resource(show_spinner="Aligning weight matrices across framework layers...")
def load_any_thesis_checkpoint(checkpoint_filename):
    """
    Natively inspects state dictionary structures and maps parameter tensors to CPU.
    Dynamically tracks keys to support standalone encoders and complete classification pipelines.
    """
    device = torch.device('cpu')
    base_encoder = SimpleEncoder()
    full_model = SSLClassifier(base_encoder, num_classes=3)
    
    if not os.path.exists(checkpoint_filename):
        return full_model, f"⚠️ Checkpoint file '{checkpoint_filename}' not found on disk. Initialized with random weights.", "warning"
        
    try:
        state_dict = torch.load(checkpoint_filename, map_location=device)
        
        # Unpack master dictionary wrappers if present
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        # Standardize DataParallel prefixes
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '')
            cleaned_state_dict[new_key] = v
            
        # Check layout properties to determine alignment path
        sample_key = next(iter(cleaned_state_dict.keys()))
        
        # Path A: File represents a standalone encoder checkpoint (e.g., ssl_encoder_real_24000.pth)
        if sample_key.startswith('features.'):
            # Map raw encoder keys directly into the sub-backbone of our classifier
            encoder_only_dict = {f"encoder.{k}": v for k, v in cleaned_state_dict.items()}
            missing_keys, unexpected_keys = full_model.load_state_dict(encoder_only_dict, strict=False)
            full_model.eval()
            return full_model, f"ℹ️ Standalone Encoder Backbone loaded. Classification head is uninitialized. (Ideal for feature exploration, less accurate for raw downstream inference).", "info"
            
        # Path B: File represents a completed fine-tuned classifier (e.g., full framework or ablations)
        else:
            full_model.load_state_dict(cleaned_state_dict, strict=True)
            full_model.eval()
            return full_model, f"✅ Clean parameter matrix mapping completed successfully for '{checkpoint_filename}'. Fully compiled for clinical inference.", "success"
            
    except Exception as e:
        # Fallback lenient mapping if structure differs
        try:
            state_dict = torch.load(checkpoint_filename, map_location=device)
            full_model.load_state_dict(state_dict, strict=False)
            full_model.eval()
            return full_model, f"⚠️ Partial loading fallback applied for '{checkpoint_filename}'. Some parameters initialized as defaults: {str(e)}", "warning"
        except Exception as crash_err:
            full_model.eval()
            return full_model, f"❌ Load Failure: Structurally incompatible dictionary file. ({str(crash_err)})", "danger"

# ==============================================================================
# 4. SIDEBAR CONFIGURATIONS & COMPREHENSIVE ABLATION BENCHMARKS
# ==============================================================================
st.sidebar.markdown("""
    <div style='text-align: center; padding: 10px; background-color: #1e3a8a; border-radius: 8px; color: white; margin-bottom: 20px;'>
        <h3 style='margin: 0; font-size: 16px;'>🔬 THESIS presentation MODE</h3>
        <span style='font-size: 11px; opacity: 0.8;'>Two-Stage Learning Evaluation Layout</span>
    </div>
""", unsafe_allow_html=True)

st.sidebar.subheader("Active Weight Vector Settings")
checkpoint_options = [
    "ssl_real_24000_finetuned.pth",
    "rebalance_only_finetuned.pth",
    "ssl_only_finetuned.pth",
    "ssl_encoder_real_24000.pth"
]

selected_checkpoint = st.sidebar.selectbox(
    "Select Model Weight Checkpoint:",
    checkpoint_options,
    help="Dynamically updates the backpropagation model layer weights for clinical inference evaluation."
)

# Initialize Selected Weight Vector
model, status_msg, status_type = load_any_thesis_checkpoint(selected_checkpoint)

# Display specific baseline performance metrics inside presentation sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Empirical Ablation Metrics")

if selected_checkpoint == "ssl_real_24000_finetuned.pth":
    st.sidebar.markdown("""
    <div class='metric-container' style='border-left: 4px solid #10b981;'>
        <b style='color: #047857;'>Full Proposed Framework (SSL + Rebalance)</b><br>
        <span style='font-size: 13px; color: #4b5563;'>Synergistic deployment leveraging domain-specific representation with class calibration.</span><br><br>
        • <b>Test Accuracy:</b> 67.3%<br>
        • <b style='color: #b91c1c;'>Melanoma Recall: 50.0% (4/8 Case Detections)</b><br>
        • <b>BKL Recall:</b> 14.3%<br>
        • <b>NV Recall:</b> 84.0%<br>
        <span style='font-size: 11px; color: #059669; font-weight: 600;'>🎯 Evaluates optimally for patient survival pathways.</span>
    </div>
    """, unsafe_allow_html=True)
elif selected_checkpoint == "rebalance_only_finetuned.pth":
    st.sidebar.markdown("""
    <div class='metric-container' style='border-left: 4px solid #f59e0b;'>
        <b style='color: #b45309;'>Ablation Variant 2: Rebalance Only (No SSL)</b><br>
        <span style='font-size: 13px; color: #4b5563;'>Randomly initialized architecture optimized solely via Focal Loss and heavy MEL oversampling.</span><br><br>
        • <b>Test Accuracy:</b> 64.4%<br>
        • <b>Melanoma Recall:</b> 37.5% (3/8 Case Detections)<br>
        • <b>BKL Recall:</b> 61.9%<br>
        • <b>NV Recall:</b> 68.0%<br>
        <span style='font-size: 11px; color: #d97706; font-weight: 600;'>⚠️ Limited resolution bounds without SSL features.</span>
    </div>
    """, unsafe_allow_html=True)
elif selected_checkpoint == "ssl_only_finetuned.pth":
    st.sidebar.markdown("""
    <div class='metric-container' style='border-left: 4px solid #ef4444;'>
        <b style='color: #b91c1c;'>Ablation Variant 1: SSL Only (No Rebalance)</b><br>
        <span style='font-size: 13px; color: #4b5563;'>Uses SimCLR weights but fine-tunes under standard cross-entropy loss boundaries.</span><br><br>
        • <b>Test Accuracy:</b> 72.1%<br>
        • <b style='color: #b91c1c;'>Melanoma Recall: 0.0% (0/8 Detected)</b><br>
        • <b>BKL Recall:</b> 5.0%<br>
        • <b>NV Recall:</b> 98.7%<br>
        <span style='font-size: 11px; color: #dc2626; font-weight: 600;'>❌ Collapses into the majority class (Nevus).</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div class='metric-container' style='border-left: 4px solid #6b7280;'>
        <b style='color: #374151;'>Raw Self-Supervised Backbone Model</b><br>
        <span style='font-size: 13px; color: #4b5563;'>Pretrained representation vectors across 24,000 unannotated dermoscopic scans using contrastive NT-Xent loss.</span><br><br>
        • <b>Pretraining Epochs:</b> 20<br>
        • <b>Batch Size Boundary:</b> 64<br>
        • <b>Backbone Latent Vector Space:</b> 128 Dim<br>
        <span style='font-size: 11px; color: #4b5563; font-weight: 600;'>ℹ️ Select a finetuned variant for downstream diagnostic scoring.</span>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("""
    <div style='background-color: #fffbeb; border: 1px solid #fde68a; padding: 12px; border-radius: 6px; font-size: 12px;'>
        <b>💡 Presentation Tip for Defense:</b> Emphasize that standard cross-entropy yields 0% Melanoma recall due to severe majority class collapse. Combining SSL representations with class rebalancing is what unlocks the clinically viable 50% target sensitivity.
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# 5. MAIN EVALUATION PLATFORM & DIAGNOSTIC INTELLIGENCE WORKSPACE
# ==============================================================================
st.markdown("""
    <div class='custom-header'>
        <h1 style='margin:0; font-size: 26px; font-weight: 700;'>Two-Stage Learning Evaluation Workspace</h1>
        <p style='margin:5px 0 0 0; opacity: 0.9; font-size: 14px;'>Interactive Research Verification Dashboard for Automated Skin Lesion Diagnosis under Small Imbalanced Data Constraints</p>
    </div>
""", unsafe_allow_html=True)

# Display Loader Status Message
if status_type == "success":
    st.markdown(f"<div class='status-badge badge-success'>{status_msg}</div><br><br>", unsafe_allow_html=True)
elif status_type == "info":
    st.markdown(f"<div class='status-badge badge-info'>{status_msg}</div><br><br>", unsafe_allow_html=True)
else:
    st.markdown(f"<div class='status-badge badge-warning'>{status_msg}</div><br><br>", unsafe_allow_html=True)

col_layout_left, col_layout_right = st.columns([1, 1])

with col_layout_left:
    st.markdown("""
        <div style='background-color: white; padding: 18px; border-radius: 8px; border: 1px solid #e2e8f0;'>
            <h3 style='margin-top:0; font-size: 18px; color: #1e3a8a;'>📷 Patient Dermoscopic Input</h3>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_lesion_file = st.file_uploader(
        "Upload localized skin scan sample (Supported extensions: PNG, JPG, JPEG):",
        type=["png", "jpg", "jpeg"],
        key="main_lesion_uploader"
    )
    
    if uploaded_lesion_file is not None:
        pil_img = Image.open(uploaded_lesion_file).convert('RGB')
        st.image(pil_img, caption="Target Dermoscopic Image Matrix (Active State)", use_column_width=True)
        
        # Consistent preprocessing logic as configured across all training stages
        preprocessing_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # Generate batch dimension [1, 3, 224, 224]
        input_tensor_batch = preprocessing_transform(pil_img).unsqueeze(0)
    else:
        st.markdown("""
            <div style='background-color: #f8fafc; border: 1px dashed #cbd5e1; padding: 40px; text-align: center; border-radius: 6px; color: #64748b; margin-top: 15px;'>
                Awaiting unannotated skin lesion upload to trigger model layer forward activation pass...
            </div>
        """, unsafe_allow_html=True)

with col_layout_right:
    st.markdown("""
        <div style='background-color: white; padding: 18px; border-radius: 8px; border: 1px solid #e2e8f0;'>
            <h3 style='margin-top:0; font-size: 18px; color: #1e3a8a;'>📊 Downstream Classification Analytics</h3>
        </div>
    """, unsafe_allow_html=True)
    
    if uploaded_lesion_file is not None:
        with st.spinner("Executing network tensor transformation pipeline..."):
            with torch.no_grad():
                network_outputs = model(input_tensor_batch)
                softmax_probabilities = torch.softmax(network_outputs, dim=1).squeeze().numpy()
                
        highest_probability_idx = softmax_probabilities.argmax()
        predicted_class_token = CLASS_NAMES[highest_probability_idx]
        confidence_metric_score = softmax_probabilities[highest_probability_idx]
        
        # Highly aesthetic metric visualization block
        st.markdown(f"""
            <div style='background-color: #ffffff; padding: 20px; border-radius: 8px; border-left: 5px solid #3b82f6; box-shadow: 0 2px 4px rgba(0,0,0,0.02); margin: 15px 0;'>
                <span style='font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase;'>Predicted Class Vector</span>
                <h2 style='margin: 4px 0 0 0; color: #1e3a8a; font-size: 24px;'>{CLASS_LABELS[predicted_class_token]}</h2>
                <div style='margin-top: 8px; font-size: 16px; font-weight: 700; color: #10b981;'>
                    {confidence_metric_score*100:.2f}% Confidence Verification Score
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h4 style='font-size: 14px; color: #475569; margin-bottom: 8px;'>Framework Probability Vector Densities:</h4>", unsafe_allow_html=True)
        for class_idx, name in enumerate(CLASS_NAMES):
            density_p = float(softmax_probabilities[class_idx])
            st.write(f"**{name}** — <span style='color:#64748b; font-size:12px;'>{CLASS_LABELS[name]}</span>", unsafe_allow_html=True)
            st.progress(density_p)
            st.caption(f"Calculated soft probability output score: {density_p*100:.2f}%")
            
        # Clinical Risk Escalation Warnings (Critical for demonstrating safety criteria)
        melanoma_risk_index = CLASS_NAMES.index('MEL')
        melanoma_calculated_score = softmax_probabilities[melanoma_risk_index]
        
        if melanoma_calculated_score > 0.15: # Critical safety flag if Melanoma activation bounds expand over 15%
            st.markdown(f"""
                <div style='background-color: #fdf2f2; border: 1px solid #fde8e8; border-left: 4px solid #ef4444; padding: 15px; border-radius: 6px; margin-top: 20px;'>
                    <h4 style='color: #9b1c1c; margin: 0 0 5px 0; font-size: 14px; font-weight: 700;'>⚠️ Critical Risk Escalation Threshold Triggered</h4>
                    <p style='color: #b91c1c; font-size: 13px; margin: 0;'>
                        Melanoma target activation score outputted at <b>{melanoma_calculated_score*100:.2f}%</b>. 
                        In order to guarantee patient survival pathways and limit false negative diagnostic exclusions (Thesis Core Metric Constraint: 50% Sensitivity), this sample should be routed for direct specialist biopsy review.
                    </p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style='color: #94a3b8; text-align: center; padding: 40px; font-size: 14px;'>
                Upload a lesion scan matrix sample in the input module to populate diagnostic analytical metrics.
            </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# 6. PIPELINE DIAGRAM SECTION (Visualizes Two-Stage Framework Mechanics)
# ==============================================================================
st.markdown("---")
st.markdown("""
    <div style='background-color: white; padding: 20px; border-radius: 8px; border: 1px solid #e2e8f0;'>
        <h3 style='margin-top:0; font-size: 18px; color: #1e3a8a; border-bottom: 2px solid #f1f5f9; padding-bottom: 10px;'>⛓️ Two-Stage Learning Pipeline Execution Flow</h3>
        <p style='font-size: 13.5px; color: #475569;'>
            This system implements the unified framework defended in your undergraduate thesis document. The pipeline follows a linear two-stage processing method to guarantee feature generalization under tight manual labeling constraints:
        </p>
        <div style='display: table; width: 100%; table-layout: fixed; margin-top: 15px; background-color: #f8fafc; padding: 15px; border-radius: 6px;'>
            <div style='display: table-cell; text-align: center; padding: 10px; border-right: 2px solid #e2e8f0;'>
                <b style='color: #1e3a8a; font-size: 14px;'>STAGE 1: Contrastive Feature Space Mapping</b><br>
                <span style='font-size: 12px; color: #64748b;'>Optimizes an unannotated pool of 24,000 real images via SimCLR transforms to extract localized edge and texture vectors without manual diagnostic labeling expense.</span>
            </div>
            <div style='display: table-cell; text-align: center; padding: 10px;'>
                <b style='color: #10b981; font-size: 14px;'>STAGE 2: Adaptive Class Rebalancing</b><br>
                <span style='font-size: 12px; color: #64748b;'>Transfers the frozen pretrained features to a classifier head, unfreezing the full network layers to adapt to 483 small labeled training images under Focal Loss calibration weights (MEL=4.0, BKL=2.0, NV=1.0).</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("<br><p style='text-align: center; color: #94a3b8; font-size: 11px;'>Thesis Evaluation System Workspace Platform • Daffodil International University • Open Source Replication Layer</p>", unsafe_allow_html=True)
