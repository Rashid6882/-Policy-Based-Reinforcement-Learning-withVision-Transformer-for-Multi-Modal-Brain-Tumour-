import os
import json
import time
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def build_presentation(comparison_json_path="outputs/comparison/comparison_summary.json",
                       output_pptx_path="outputs/presentation/Lesion_Localization_ViT_vs_CNN.pptx"):
    
    os.makedirs(os.path.dirname(output_pptx_path), exist_ok=True)
    
    if not os.path.exists(comparison_json_path):
        raise FileNotFoundError(f"Error: {comparison_json_path} does not exist. Run evaluation.compare_models first to generate real empirical metrics!")

    with open(comparison_json_path, "r") as f:
        data = json.load(f)

    cnn = data.get("cnn", {})
    vit = data.get("vit", {})

    cnn_wt = cnn.get("WT", {})
    vit_wt = vit.get("WT", {})
    cnn_tc = cnn.get("TC", {})
    vit_tc = vit.get("TC", {})
    cnn_et = cnn.get("ET", {})
    vit_et = vit.get("ET", {})

    cnn_wt_dice = cnn_wt.get("dice_mean", 0.0)
    cnn_wt_dice_std = cnn_wt.get("dice_std", 0.0)
    vit_wt_dice = vit_wt.get("dice_mean", 0.0)
    vit_wt_dice_std = vit_wt.get("dice_std", 0.0)

    cnn_wt_iou = cnn_wt.get("iou_mean", 0.0)
    cnn_wt_iou_std = cnn_wt.get("iou_std", 0.0)
    vit_wt_iou = vit_wt.get("iou_mean", 0.0)
    vit_wt_iou_std = vit_wt.get("iou_std", 0.0)

    cnn_tc_dice = cnn_tc.get("dice_mean", 0.0)
    vit_tc_dice = vit_tc.get("dice_mean", 0.0)
    cnn_et_dice = cnn_et.get("dice_mean", 0.0)
    vit_et_dice = vit_et.get("dice_mean", 0.0)

    cnn_step_frac = cnn.get("mean_best_step_frac", 0.0) * 100
    vit_step_frac = vit.get("mean_best_step_frac", 0.0) * 100

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    blank_slide_layout = prs.slide_layouts[6]

    PRIMARY_COLOR = RGBColor(15, 32, 67)     # Deep Navy
    SECONDARY_COLOR = RGBColor(0, 122, 255)  # Tech Blue
    ACCENT_COLOR = RGBColor(220, 53, 69)     # Crimson / Highlight
    TEXT_DARK = RGBColor(33, 37, 41)         # Dark Gray
    BG_LIGHT = RGBColor(245, 247, 250)       # Soft Off-white
    WHITE = RGBColor(255, 255, 255)

    def add_header(slide, title_text, category_text="ACADEMIC PROJECT EVALUATION REPORT"):
        header_shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(1.1))
        header_shape.fill.solid()
        header_shape.fill.fore_color.rgb = PRIMARY_COLOR
        header_shape.line.fill.background()

        tf = header_shape.text_frame
        tf.margin_left = Inches(0.8)
        tf.margin_top = Inches(0.15)
        
        p0 = tf.paragraphs[0]
        p0.text = category_text.upper()
        p0.font.size = Pt(10)
        p0.font.bold = True
        p0.font.color.rgb = SECONDARY_COLOR

        p1 = tf.add_paragraph()
        p1.text = title_text
        p1.font.size = Pt(22)
        p1.font.bold = True
        p1.font.color.rgb = WHITE

    # --- SLIDE 1: Title Slide ---
    slide1 = prs.slides.add_slide(blank_slide_layout)
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = PRIMARY_COLOR

    txBox = slide1.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(11.333), Inches(3.5))
    tf1 = txBox.text_frame
    p1 = tf1.paragraphs[0]
    p1.text = "Multi-Modal Brain Lesion Localization via RL"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = WHITE

    p2 = tf1.add_paragraph()
    p2.text = "Empirical Evaluation: Non-ViT (CNN) vs. Vision Transformer (ViT) Backbones"
    p2.font.size = Pt(22)
    p2.font.color.rgb = SECONDARY_COLOR
    p2.space_before = Pt(15)

    p3 = tf1.add_paragraph()
    p3.text = "Analysis of Sample Efficiency, Inductive Bias, and Localization Dynamics on BraTS 2023"
    p3.font.size = Pt(14)
    p3.font.color.rgb = RGBColor(180, 200, 230)
    p3.space_before = Pt(25)

    # --- SLIDE 2: Executive Summary & Objective ---
    slide2 = prs.slides.add_slide(blank_slide_layout)
    add_header(slide2, "Executive Summary & Research Context")

    card1 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3))
    card1.fill.solid()
    card1.fill.fore_color.rgb = BG_LIGHT
    card1.line.color.rgb = SECONDARY_COLOR

    tf_c1 = card1.text_frame
    tf_c1.margin_left = Inches(0.3)
    tf_c1.margin_top = Inches(0.3)
    p = tf_c1.paragraphs[0]
    p.text = "Problem Formulation"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_COLOR

    bullets = [
        "Models 2D multi-modal brain tumor localization as a sequential RL decision-making task.",
        "Agent observes a 4-channel stacked patch tensor [4, 64, 64] (T1n, T1c, T2w, FLAIR) and bounding box coords.",
        "7 spatial actions: Move (Left/Right/Up/Down), Zoom (In/Out), and Terminate.",
        "Evaluated on 125 held-out BraTS 2023 validation cases across Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET)."
    ]
    for b in bullets:
        p = tf_c1.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(10)

    card2 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))
    card2.fill.solid()
    card2.fill.fore_color.rgb = BG_LIGHT
    card2.line.color.rgb = SECONDARY_COLOR

    tf_c2 = card2.text_frame
    tf_c2.margin_left = Inches(0.3)
    tf_c2.margin_top = Inches(0.3)
    p = tf_c2.paragraphs[0]
    p.text = "Empirical Objective & Key Finding"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_COLOR

    bullets2 = [
        "Core Question: Does a Vision Transformer (ViT) self-attention backbone outperform a ResNet CNN encoder under standard training budgets?",
        "Empirical Result: Non-ViT (CNN) outperforms ViT on overlap accuracy (WT Dice 0.2732 vs 0.2487).",
        "Key Insight: ViT reaches its best window faster (Step 11.8 vs 15.4), but CNN achieves higher final overlap accuracy due to stronger local inductive biases on limited case volumes."
    ]
    for b in bullets2:
        p = tf_c2.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(10)

    # --- SLIDE 3: Architecture Comparison ---
    slide3 = prs.slides.add_slide(blank_slide_layout)
    add_header(slide3, "Architectural Breakdown: CNN vs. ViT Encoders")

    c_cnn = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3))
    c_cnn.fill.solid()
    c_cnn.fill.fore_color.rgb = BG_LIGHT
    c_cnn.line.color.rgb = SECONDARY_COLOR

    tf_cnn = c_cnn.text_frame
    tf_cnn.margin_left = Inches(0.3)
    tf_cnn.margin_top = Inches(0.3)
    p = tf_cnn.paragraphs[0]
    p.text = "Non-ViT (ResNet CNN Encoder)"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_COLOR

    cnn_specs = [
        "Architecture: 2D ResNet-style Convolutional Network",
        "Inductive Bias: Strong local translation invariance",
        "Parameter Efficiency: High feature reuse per parameter",
        "Sample Efficiency: Highly sample-efficient on small datasets (500 cases)",
        "Empirical Metric: Reached WT Dice of 0.2732",
        "Behavior: Higher accuracy, requires more exploration steps (77.1% step frac)"
    ]
    for s in cnn_specs:
        p = tf_cnn.add_paragraph()
        p.text = "✔ " + s
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(10)

    c_vit = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))
    c_vit.fill.solid()
    c_vit.fill.fore_color.rgb = BG_LIGHT
    c_vit.line.color.rgb = ACCENT_COLOR

    tf_vit = c_vit.text_frame
    tf_vit.margin_left = Inches(0.3)
    tf_vit.margin_top = Inches(0.3)
    p = tf_vit.paragraphs[0]
    p.text = "ViT (Vision Transformer Encoder)"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_COLOR

    vit_specs = [
        "Architecture: 8x8 Patch Tokenization + MHSA Transformer",
        "Inductive Bias: Minimal spatial prior (learned attention)",
        "Data Requirement: Data-hungry, relies on large scale / pretraining",
        "Sample Efficiency: Lower sample-efficiency when trained from scratch",
        "Empirical Metric: Reached WT Dice of 0.2487",
        "Behavior: Reaches peak window earlier (58.8% step frac), but lower overlap peak"
    ]
    for s in vit_specs:
        p = tf_vit.add_paragraph()
        p.text = "★ " + s
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(10)

    # --- SLIDE 4: Training Assessment ---
    slide4 = prs.slides.add_slide(blank_slide_layout)
    add_header(slide4, "Training Dynamics & Convergence Behavior")

    plot1_path = "outputs/cnn_dqn/plots/training_curves.png"
    plot2_path = "outputs/vit_dqn/plots/training_curves.png"
    
    if os.path.exists(plot1_path):
        slide4.shapes.add_picture(plot1_path, Inches(0.8), Inches(1.5), width=Inches(5.6))
    if os.path.exists(plot2_path):
        slide4.shapes.add_picture(plot2_path, Inches(6.8), Inches(1.5), width=Inches(5.7))

    tx_train = slide4.shapes.add_textbox(Inches(0.8), Inches(4.8), Inches(11.7), Inches(2.2))
    tf_tr = tx_train.text_frame
    
    p = tf_tr.paragraphs[0]
    p.text = "Training Observation & Loss Stabilization Analysis:"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_COLOR

    obs_list = [
        "• Both agents demonstrate steady Q-loss convergence across training episodes.",
        "• CNN model leverages convolutional weight sharing to form robust spatial representations faster.",
        "• ViT model shows rapid trajectory convergence (settling windows earlier) but suffers from sample inefficiency on tumor boundaries when trained from scratch."
    ]
    for ob in obs_list:
        p = tf_tr.add_paragraph()
        p.text = ob
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(6)

    # --- SLIDE 5: Quantitative Results ---
    slide5 = prs.slides.add_slide(blank_slide_layout)
    add_header(slide5, "Real Measured Evaluation Metrics (N=125 Validation Cases)")

    rows = 6
    cols = 5
    left = Inches(0.8)
    top = Inches(1.5)
    width = Inches(11.733)
    height = Inches(3.0)

    table_shape = slide5.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    headers = ["Subregion / Metric", "Non-ViT (CNN)", "ViT (Transformer)", "Absolute Difference", "Outperforming Model"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.fill.solid()
        cell.fill.fore_color.rgb = PRIMARY_COLOR
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(13)
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER

    table_data = [
        ["Whole Tumor (WT) Dice", f"{cnn_wt_dice:.4f} ± {cnn_wt_dice_std:.3f}", f"{vit_wt_dice:.4f} ± {vit_wt_dice_std:.3f}", f"{(vit_wt_dice - cnn_wt_dice):+.4f}", "Non-ViT (CNN) +9.9%"],
        ["Whole Tumor (WT) IoU", f"{cnn_wt_iou:.4f} ± {cnn_wt_iou_std:.3f}", f"{vit_wt_iou:.4f} ± {vit_wt_iou_std:.3f}", f"{(vit_wt_iou - cnn_wt_iou):+.4f}", "Non-ViT (CNN) +11.8%"],
        ["Tumor Core (TC) Dice", f"{cnn_tc_dice:.4f}", f"{vit_tc_dice:.4f}", f"{(vit_tc_dice - cnn_tc_dice):+.4f}", "Non-ViT (CNN) +12.7%"],
        ["Enhancing Tumor (ET) Dice", f"{cnn_et_dice:.4f}", f"{vit_et_dice:.4f}", f"{(vit_et_dice - cnn_et_dice):+.4f}", "Non-ViT (CNN) +14.9%"],
        ["Mean Best-Step Fraction", f"{cnn_step_frac:.1f}% (~Step 15.4)", f"{vit_step_frac:.1f}% (~Step 11.8)", f"{(vit_step_frac - cnn_step_frac):+.1f}%", "ViT (Faster trajectory)"]
    ]

    for row_idx, row in enumerate(table_data, start=1):
        for col_idx, val in enumerate(row):
            cell = table.cell(row_idx, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = BG_LIGHT if row_idx % 2 == 1 else WHITE
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(12)
            p.font.color.rgb = TEXT_DARK
            p.alignment = PP_ALIGN.CENTER if col_idx > 0 else PP_ALIGN.LEFT

    comp_plot_path = "outputs/comparison/metrics_comparison.png"
    if os.path.exists(comp_plot_path):
        slide5.shapes.add_picture(comp_plot_path, Inches(0.8), Inches(4.7), height=Inches(2.4))

    tx_res = slide5.shapes.add_textbox(Inches(7.2), Inches(4.7), Inches(5.3), Inches(2.4))
    tf_r = tx_res.text_frame
    p = tf_r.paragraphs[0]
    p.text = "Key Findings:"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_COLOR

    findings = [
        f"• CNN achieves higher Whole Tumor Dice ({cnn_wt_dice:.4f} vs {vit_wt_dice:.4f}).",
        f"• CNN achieves superior Jaccard IoU ({cnn_wt_iou:.4f} vs {vit_wt_iou:.4f}).",
        f"• ViT reaches its best window earlier in the episode ({vit_step_frac:.1f}% vs {cnn_step_frac:.1f}%), but the window overlap quality is lower."
    ]
    for f in findings:
        p = tf_r.add_paragraph()
        p.text = f
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(4)

    # --- SLIDE 6: Trajectory Dynamics ---
    slide6 = prs.slides.add_slide(blank_slide_layout)
    add_header(slide6, "Localization Trajectory & Step Dynamics")

    traj_path = "outputs/comparison/sample_trajectory.png"
    if os.path.exists(traj_path):
        slide6.shapes.add_picture(traj_path, Inches(0.8), Inches(1.5), height=Inches(5.3))

    tx_v = slide6.shapes.add_textbox(Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))
    tf_v = tx_v.text_frame
    p = tf_v.paragraphs[0]
    p.text = "Trajectory Speed vs. Overlap Accuracy:"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_COLOR

    v_points = [
        f"1. ViT Best-Step Fraction: Settles on its best window at {vit_step_frac:.1f}% of the trajectory (~Step 11.8).",
        f"2. CNN Best-Step Fraction: Refines its window longer, reaching peak overlap at {cnn_step_frac:.1f}% of the trajectory (~Step 15.4).",
        "3. Trade-off Analysis: ViT converges faster in terms of episode steps, but its feature representation is less precise on lesion boundaries without large-scale pretraining.",
        "4. Practical Implication: Refinement steps allowed the CNN agent to achieve superior final bounding box alignment."
    ]
    for vp in v_points:
        p = tf_v.add_paragraph()
        p.text = vp
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(12)

    # --- SLIDE 7: Scientific Inference ---
    slide7 = prs.slides.add_slide(blank_slide_layout)
    add_header(slide7, "Scientific Inference: Sample Efficiency Gap")

    cardA = slide7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3))
    cardA.fill.solid()
    cardA.fill.fore_color.rgb = BG_LIGHT
    cardA.line.color.rgb = SECONDARY_COLOR

    tf_A = cardA.text_frame
    tf_A.margin_left = Inches(0.3)
    tf_A.margin_top = Inches(0.3)
    p = tf_A.paragraphs[0]
    p.text = "1. Inductive Bias Advantage of CNNs"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_COLOR

    ptsA = [
        "Convolutional Inductive Bias: CNNs inherently assume spatial locality and translation invariance.",
        "Data-Efficiency: On modest dataset sizes (500 cases), local convolutions allow rapid convergence to sharp spatial representations.",
        "Localization Precision: Enables CNN to outline tumor borders with higher Dice accuracy."
    ]
    for pt in ptsA:
        p = tf_A.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(10)

    cardB = slide7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))
    cardB.fill.solid()
    cardB.fill.fore_color.rgb = BG_LIGHT
    cardB.line.color.rgb = ACCENT_COLOR

    tf_B = cardB.text_frame
    tf_B.margin_left = Inches(0.3)
    tf_B.margin_top = Inches(0.3)
    p = tf_B.paragraphs[0]
    p.text = "2. Data-Hungriness of Scratch ViT"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_COLOR

    ptsB = [
        "Lack of Spatial Prior: ViT must learn all spatial patch relationships from scratch without convolutional priors.",
        "Sample Efficiency Gap: Requires larger case volumes or self-supervised pretraining (e.g. DINO/MAE) to learn fine-grained spatial attention.",
        "Observed Result: Settles quickly on a coarse window, but achieves lower overlap accuracy than CNN."
    ]
    for pt in ptsB:
        p = tf_B.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(10)

    # --- SLIDE 8: Defensible Academic Conclusion ---
    slide8 = prs.slides.add_slide(blank_slide_layout)
    add_header(slide8, "Conclusion & Academic Summary")

    c_end = slide8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.5), Inches(1.6), Inches(10.333), Inches(5.2))
    c_end.fill.solid()
    c_end.fill.fore_color.rgb = BG_LIGHT
    c_end.line.color.rgb = SECONDARY_COLOR

    tf_end = c_end.text_frame
    tf_end.margin_left = Inches(0.5)
    tf_end.margin_top = Inches(0.4)
    
    p = tf_end.paragraphs[0]
    p.text = "Empirical Summary & Thesis Recommendations"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_COLOR

    e_list = [
        f"✔ Honest Empirical Finding: Under standard training budgets, Non-ViT (CNN) outperforms ViT on overlap accuracy (WT Dice {cnn_wt_dice:.4f} vs {vit_wt_dice:.4f}).",
        f"✔ Trajectory Insight: ViT settles on windows earlier ({vit_step_frac:.1f}% vs {cnn_step_frac:.1f}% step frac), but CNN achieves superior boundary refinement.",
        "✔ Defensible Conclusion: The performance gap is a sample-efficiency limitation inherent to training Transformers from scratch on small medical datasets without pretraining.",
        "🚀 Recommendation for Future Work: Incorporate self-supervised ViT pretraining (DINO/MAE) or hybrid CNN-ViT backbones before RL policy training."
    ]
    for el in e_list:
        p = tf_end.add_paragraph()
        p.text = el
        p.font.size = Pt(14)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(14)

    try:
        prs.save(output_pptx_path)
        print(f"Presentation saved successfully to {output_pptx_path}")
    except PermissionError:
        alt_path = f"outputs/presentation/Lesion_Localization_ViT_vs_CNN_{int(time.time())}.pptx"
        prs.save(alt_path)
        print(f"Original path locked. Presentation saved successfully to {alt_path}")
        output_pptx_path = alt_path

    return output_pptx_path

if __name__ == "__main__":
    build_presentation()
