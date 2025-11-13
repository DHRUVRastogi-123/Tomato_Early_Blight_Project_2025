# SLIDE-BY-SLIDE VISUAL & CONTENT CHECKLIST

## A Practical Guide for Building Each Slide

---

## SLIDE 1: TITLE SLIDE ✓

### Visual Layout:
```
┌─────────────────────────────────────────────┐
│                                             │
│     [HEALTHY LEAF]    [DISEASED LEAF]      │
│         (60%)              (40%)            │
│                                             │
│     Title: Tomato Plant Early Blight       │
│            Disease Detection Model         │
│                                             │
│     Team: Dhruv, Himanshu, Dheeraj, Neha   │
│     Supervised by: Dr. J Sathish Kumar     │
│     MNNIT Allahabad                        │
│                                             │
└─────────────────────────────────────────────┘
```

### Image Specs:
- **Left image:** Vibrant green healthy leaf, close-up, high contrast, ~600×400px
- **Right image:** Diseased leaf with brown/yellow lesions, ~600×400px
- **Separator:** Subtle white or gradient line between images

### Text Formatting:
- Title: 54pt bold, white text on dark background (navy blue or black)
- Team names: 28pt, light gray
- Supervisor: 20pt, light gray

### Speaker Setup:
- Stand center, facing audience
- Let slide do the talking (30 seconds max)
- Make eye contact while slide is displayed

---

## SLIDE 2: PROBLEM STATEMENT ✓

### Visual Layout:
```
┌──────────────────────────────────────────────────┐
│  [FARMER INSPECTING]    [DISEASE PROGRESSION]   │
│      (Left 50%)              (Right 50%)        │
│                                                  │
│  • Manual inspection = time-consuming          │
│  • Subjective & inconsistent                   │
│  • Lack of rural expertise                     │
│  • Delayed detection → crop loss               │
│                                                  │
└──────────────────────────────────────────────────┘
```

### Image Specs:
- **Left:** Farmer with tablet/clipboard in field, realistic photography, warm lighting
- **Right:** 3-stage progression of disease on leaf (Stage 1: small spots, Stage 2: lesions, Stage 3: severe damage)
  - Use arrows or timeline format between stages
  - Color progression: yellow→brown→dark brown

### Text Formatting:
- Bullets: 32pt, dark gray
- Leave 40% of slide for text, 60% for images

### Design Notes:
- Use a timeline arrow showing disease progression
- Add subtle icons: stopwatch icon for "time-consuming," person icon for "lack of expertise"

---

## SLIDE 3: BACKGROUND & CHALLENGES ✓

### Visual Layout:
```
┌─────────────────────────────────────────────┐
│                                             │
│  ┌─────────┬─────────┬─────────┬────────┐  │
│  │ Clutter │ Shadows │Overlap  │ Stems  │  │
│  │ Image   │ Image   │ Image   │ Image  │  │
│  └─────────┴─────────┴─────────┴────────┘  │
│                                             │
│  • Background clutter confuses models      │
│  • Lighting variations (sun, weather)      │
│  • Leaf occlusion (overlapping)            │
│  • Model learns spurious features          │
│                                             │
└─────────────────────────────────────────────┘
```

### Image Specs:
- 2×2 or 2×3 grid of field images
- Each showing a different challenge
- Add red circles/arrows highlighting problem areas
- Use consistent framing for each sub-image (~300×250px each)

### Annotations:
- Label each quadrant: "Clutter," "Shadows," "Occlusion," "Variable Lighting"
- Add red warning icons where problems exist

### Design Notes:
- Use a darker background to make images pop
- Add a subtle grid separator between images

---

## SLIDE 4: PROBLEM STATEMENT & SOLUTION ✓

### Visual Layout:
```
┌──────────────────────────────┬──────────────────────────────┐
│  PROBLEM (Left 50%)          │  SOLUTION (Right 50%)         │
│  ┌─────────────────────────┐ │ ┌──────────────────────────┐ │
│  │ [Messy Field Image]     │ │ │ [Same image transformed] │ │
│  │ - Background clutter    │ │ │ ✓ Clean leaves isolated  │ │
│  │ - Lighting variation    │ │ │ ✓ No background noise    │ │
│  │ - Multiple overlaps     │ │ │ ✓ Stage 1: Segment       │ │
│  │ - Confuses classifiers  │ │ │ ✓ Stage 2: Classify      │ │
│  └─────────────────────────┘ │ └──────────────────────────┘ │
│  ❌ Standard CNN fails        │  ✅ Two-stage pipeline      │
│                               │                              │
└──────────────────────────────┴──────────────────────────────┘
```

### Image Specs:
- Left side: Raw, cluttered field image
- Right side: Same image after segmentation and cleaning (isolated leaves visible, background gone)
- Use a **large transformation arrow** between them

### Text Formatting:
- Problem bullets: Red text, 28pt
- Solution bullets: Green text, 28pt
- Use checkmarks (✓) and X marks (❌) for visual emphasis

### Design Notes:
- Create strong visual contrast between left (chaotic) and right (clean)
- Use color psychology: red for problem area, green for solution area

---

## SLIDE 5: RELATED WORK ✓

### Visual Layout:
```
┌──────────────────────────────────────────────┐
│  Existing Approaches Comparison             │
│                                              │
│  ┌─ IoT Soil Sensing ───→ 95% accuracy      │
│  │  ✓ Good precision    ✗ Can't see disease │
│  │                                          │
│  ├─ Older YOLO ────────→ 90% accuracy       │
│  │  ✓ Fast             ✗ Fails w/ clutter   │
│  │                                          │
│  ├─ Hyperspectral ─────→ 98% accuracy       │
│  │  ✓ Very accurate    ✗ Too expensive      │
│  │                                          │
│  └─ OUR APPROACH ──────→ ~97% accuracy      │
│     ✓ Practical & Deployable ✓ Accurate    │
│                                              │
└──────────────────────────────────────────────┘
```

### Image Specs:
- Small icons for each approach (soil sensor, YOLO logo, HSI camera, phone)
- Accuracy bars showing comparative performance
- Green highlight on your approach

### Text Formatting:
- Comparison points: 24pt
- Accuracy percentages: 32pt bold
- Icons: 80×80px each

### Design Notes:
- Create a visual comparison that makes your approach stand out
- Use color: gray for competitors, bright color for your work

---

## SLIDE 6: DATA SOURCES & FILTERING ✓

### Visual Layout:
```
┌──────────────────────────────────────┐
│  Data Pipeline                       │
│                                      │
│  PlantVillage  Kaggle  Mendeley     │
│       ↓           ↓         ↓        │
│       └─────────┬─────────┘          │
│             ↓   ↓   ↓                │
│          FUNNEL                      │
│             ↓                        │
│    Multi-class Dataset               │
│    (15 diseases)                     │
│             ↓                        │
│       AUTOMATED FILTER               │
│             ↓                        │
│  Binary Dataset                      │
│  Class 1: Early Blight               │
│  Class 7: Healthy                    │
│                                      │
└──────────────────────────────────────┘
```

### Image Specs:
- 3 source logos at top
- Funnel shape in center
- Multi-class dataset representation (colored squares showing 15 classes)
- Arrow down to binary output (2 colored squares: orange for Early Blight, green for Healthy)

### Text Formatting:
- Source names: 24pt
- Class names: 28pt bold
- Numbers: 32pt for class count

### Design Notes:
- Use the funnel metaphor visually
- Use color coding consistently: orange for Early Blight throughout all slides, green for Healthy

---

## SLIDE 7: RAW DATA CHALLENGES ✓

### Visual Layout:
```
┌────────────────────────────────────────┐
│  Real-World Data: The Challenge        │
│                                        │
│  [Image 1]  [Image 2]  [Image 3]      │
│  Clutter    Shadows    Occlusion       │
│                                        │
│  [Image 4]  [Image 5]  [Image 6]      │
│  Lighting   Stems      Multiple        │
│  Variation  Attached    Leaves         │
│                                        │
│  Red circles/arrows pointing to        │
│  specific problems in each image       │
│                                        │
└────────────────────────────────────────┘
```

### Image Specs:
- 3×2 grid of field images (each ~250×200px)
- Each showing a different real-world challenge
- Red circles highlighting the specific problem
- Each labeled clearly

### Annotations:
- Add labels: "Background Clutter," "Shadows," "Occlusion," "Variable Lighting," "Stems Attached," "Multiple Leaves"

### Design Notes:
- Keep images raw and realistic (no heavy filtering)
- Use consistent red circles/arrows for highlighting
- No text on images themselves; labels outside/below

---

## SLIDE 8: YOLOV8 SOLUTION ✓

### Visual Layout:
```
┌────────────────────────────────────────┐
│  Stage 1: Leaf Segmentation            │
│                                        │
│  RAW IMAGE                             │
│      ↓                                 │
│  [YOLOv8-seg Processing Icon]          │
│      ↓                                 │
│  SEGMENTATION MASKS                    │
│  (White leaves on black background)    │
│      ↓                                 │
│  ISOLATED CLEAN LEAVES                 │
│                                        │
│  Key Benefits:                         │
│  ✓ Pixel-perfect masks                 │
│  ✓ Real-time processing                │
│  ✓ 20% better than YOLOv5             │
│                                        │
└────────────────────────────────────────┘
```

### Image Specs:
- Top: Raw messy field image (400×300px)
- Middle: Processing indicator (animated or icon)
- Middle: Segmentation masks (white mask on black, 400×300px)
- Bottom: Isolated clean leaves in grid (4-6 leaves, each ~150×150px)

### Design Notes:
- Use arrows to show progression
- Color code: raw = colorful, masks = white/black, clean = cropped leaves with borders
- Add a neural network icon for processing step

---

## SLIDE 9: FROM RAW TO CLEAN ✓

### Visual Layout:
```
┌─────────────────┬──────────────────────┐
│ BEFORE (Left)   │ AFTER (Right)        │
│                 │                      │
│ [Messy Image]   │ [Clean isolated      │
│                 │  leaves grid]        │
│ • Clutter       │ ✓ Clean             │
│ • Noise         │ ✓ Isolated          │
│ • Complex       │ ✓ Standardized      │
│ • Confusing     │ ✓ Ready for ML      │
│                 │                      │
│ ========== TRANSFORMATION ==========   │
│ 1 messy image → 5 clean leaves        │
│                 │                      │
└─────────────────┴──────────────────────┘
```

### Image Specs:
- Left: Raw cluttered image (400×400px)
- Large arrow or transformation indicator (centered, ~200px wide)
- Right: Grid of 5 clean isolated leaf crops (each ~120×120px)

### Design Notes:
- Make the transformation visually dramatic
- Use fade-out effect on left image to show what's being removed
- Color the right side leaves with thin borders to show they're isolated

---

## SLIDE 10: YOLOV8 TECHNICAL DETAILS ✓

### Visual Layout:
```
┌────────────────────────────────────────┐
│  YOLOv8-seg: How It Works              │
│                                        │
│  BOUNDING BOXES              SEGMENTATION MASKS
│  ┌──────────────┐           ┌──────────────┐
│  │[Box 1]       │           │ 🔴 Leaf 1    │
│  │  [Box 2]     │           │   🟢 Leaf 2  │
│  │ [Box 3]      │           │ 🔵 Leaf 3    │
│  │     [Box 4]  │           │ 🟡 Leaf 4    │
│  └──────────────┘           └──────────────┘
│       Detection Only        Instance Segmentation
│                                        │
│  • Instance segmentation = pixel-level precision
│  • Each leaf gets unique mask (not class label)
│  • Real-time inference capability
│  • 20% accuracy improvement over YOLOv5
│                                        │
└────────────────────────────────────────┘
```

### Image Specs:
- Left: Raw image with bounding boxes (different colors for each)
- Right: Same image with segmentation masks overlaid (each leaf a different color)
- Clear visual difference between the two approaches

### Design Notes:
- Use 4 distinct colors (red, green, blue, yellow) for masks
- Show semi-transparent overlay for mask visualization
- Add small icons: box icon for detection, mask icon for segmentation

---

## SLIDE 11: DATA CHALLENGES ADDRESSED ✓

### Visual Layout:
```
┌──────────────────────────────────────┐
│  Data Engineering: Challenges Solved  │
│                                      │
│  1️⃣ Multiple Leaves               │
│     ✓ Instance segmentation          │
│                                      │
│  2️⃣ Complex Backgrounds           │
│     ✓ Mask refinement                │
│                                      │
│  3️⃣ Inconsistent Annotations      │
│     ✓ Format unification (XML→YOLO) │
│                                      │
│  4️⃣ Multiple Data Sources         │
│     ✓ Automated pipeline             │
│                                      │
└──────────────────────────────────────┘
```

### Image Specs:
- 4 quadrants, each with:
  - Challenge image/icon
  - Problem description
  - Solution with checkmark

### Design Notes:
- Use numbered circles (1️⃣, 2️⃣, etc.) for visual hierarchy
- Color each quadrant slightly differently (light backgrounds)
- Use checkmarks in green

---

## SLIDE 12: DATA AUGMENTATION ✓

### Visual Layout:
```
┌──────────────────────────────────────┐
│  Augmentation: Teaching Robustness    │
│                                      │
│        Original                      │
│          ↓                           │
│  ┌───┬───┬───┐                       │
│  │ 🔄 Flip │ Rotate │ Color Jitter  │
│  ├───┼───┼───┤                       │
│  │Horiz Vert  45°   Brightness     │
│  └───┴───┴───┘                       │
│                                      │
│  Effect: 8-10× more training data    │
│  Result: Robustness to real-world    │
│                                      │
└──────────────────────────────────────┘
```

### Image Specs:
- Center: Original leaf image (200×200px)
- Around it: 8 variations (150×150px each in 3×3 or 2×4 grid)
- Each variation clearly labeled
- Subtle visual difference between variations

### Design Notes:
- Arrange as a grid with original in prominent position
- Use subtle borders to separate variations
- Keep color consistent across all variations (showing transformation type, not content change)

---

## SLIDE 13: MODEL SELECTION ✓

### Visual Layout:
```
┌──────────────────────────────────────┐
│  Why EfficientNet-B0?                │
│                                      │
│  VGG16         ResNet-50   EfficientNet-B0
│  │             │            │
│  │ 138M params │ 25.5M      │ 5.3M
│  │ (Tall)      │ (Medium)   │ (Small)
│  │ Slow        │ Good       │ ⚡Fast
│  │ ✗Deploy     │ ~Deploy    │ ✅Deploy
│  │             │            │
│  └─────────────┴────────────┘
│                                      │
│  State-of-art accuracy + edge devices│
│  = Best choice for practical farming │
│                                      │
└──────────────────────────────────────┘
```

### Image Specs:
- Three vertical bars/silhouettes of decreasing size (representing parameter count)
- Size proportional to number of parameters
- Icons: Raspberry Pi, Jetson Nano, phone showing deployment capability

### Text Formatting:
- Model names: 28pt bold
- Parameter counts: 32pt, color-coded (red=many, yellow=medium, green=few)
- Emoji or symbols for comparison

### Design Notes:
- Make parameter size difference visually obvious
- Use green checkmark for EfficientNet, red X for others (regarding deployment)

---

## SLIDE 14: ARCHITECTURE OVERVIEW ✓

### Visual Layout:
```
┌────────────────────────────────────────┐
│  EfficientNet-B0 Architecture          │
│                                        │
│  INPUT: 224×224×3                     │
│      ↓                                │
│  [Conv] → [MBConv blocks] → [Pool]   │
│  Convolution + MobileInvertedBottle   │
│  with skip connections                │
│      ↓                                │
│  [Dense 256] → [Dense 2]             │
│      ↓                                │
│  OUTPUT: [Healthy, Early Blight]     │
│                                        │
│  Features: Depthwise-separable conv   │
│            Skip connections           │
│            Efficient scaling          │
│                                        │
└────────────────────────────────────────┘
```

### Image Specs:
- Simplified architecture diagram (not too technical)
- Flow arrows showing data progression
- Layer blocks in different colors (blue=conv, green=pooling, orange=FC)
- Spatial dimensions decreasing as you go deeper (visual pyramid)

### Design Notes:
- Use color to distinguish layer types
- Show one or two skip connections as curved lines
- Include spatial dimensions (224×224 → 112×112 → 56×56, etc.)

---

## SLIDE 15: TRANSFER LEARNING ✓

### Visual Layout:
```
┌──────────────────────────────────────┐
│  Transfer Learning: Standing on       │
│  the Shoulders of Giants              │
│                                      │
│  ImageNet Pre-training                │
│  ↓                                    │
│  [Diverse images: animals, cars,     │
│   landscapes, furniture...]           │
│  1.2M images, 1000 classes           │
│  ↓                                    │
│  Model learns: edges, textures,      │
│  colors, shapes                       │
│  ↓                                    │
│  Transfer to Our Task                 │
│  ↓                                    │
│  Replace final layer: 1000→2 classes │
│  Fine-tune with low learning rate    │
│  ↓                                    │
│  Result: Fast training, fewer data   │
│                                      │
└──────────────────────────────────────┘
```

### Image Specs:
- Top: ImageNet collage (4-6 diverse images representing different classes)
- Middle: Model weights visualization (heatmap or matrix)
- Bottom: Our dataset (tomato leaves)
- Arrows connecting each step

### Design Notes:
- Show contrast between ImageNet diversity and our specific task
- Use blue color for ImageNet, green for our task
- Show layer freeze status: blue (frozen) vs. orange (trainable)

---

## SLIDE 16: TRAINING STRATEGY ✓

### Visual Layout:
```
┌────────────────────────────────────────┐
│  Training Configuration & Results      │
│                                        │
│  Hyperparameters:                      │
│  • Batch size: 32                      │
│  • Learning rate: 0.001                │
│  • Optimizer: Adam                     │
│  • Loss function: Cross-entropy        │
│  • Epochs: 25                          │
│  • Augmentation: Active                │
│                                        │
│  Learning Curves:                      │
│  [Graph showing training and           │
│   validation loss converging]          │
│                                        │
│  Observation: Minimal overfitting      │
│  ✓ Model generalizes well             │
│                                        │
└────────────────────────────────────────┘
```

### Image Specs:
- Left (50%): Table of hyperparameters
- Right (50%): Learning curves graph
  - X-axis: Epochs (0-25)
  - Y-axis: Loss (0-0.5)
  - Blue line: Training loss
  - Orange line: Validation loss
  - Both curves converging

### Design Notes:
- Use a clean table format for hyperparameters
- Graph should show clear convergence (not divergence)
- Add annotations pointing to key observations

---

## SLIDE 17: RESULTS - ACCURACY METRICS ✓

### Visual Layout:
```
┌────────────────────────────────────────┐
│  Classification Performance            │
│                                        │
│  ┌─────────────────────┐              │
│  │ Accuracy: 97.4%     │ ▓▓▓▓▓▓▓▓▓█  │
│  │ Precision: 96.8%    │ ▓▓▓▓▓▓▓▓░   │
│  │ Recall: 98.2%       │ ▓▓▓▓▓▓▓▓▓  │
│  │ F1-Score: 97.5%     │ ▓▓▓▓▓▓▓▓█  │
│  └─────────────────────┘              │
│                                        │
│  Interpretation:                      │
│  • 97 out of 100 leaves correct       │
│  • False positives: Low (rare)        │
│  • False negatives: Very low (catch   │
│    98% of actual diseases)            │
│                                        │
└────────────────────────────────────────┘
```

### Image Specs:
- Four horizontal progress bars (red→green gradient)
- Each bar reaching 96-98%
- Numeric value on the right of each bar
- Green checkmark for excellent performance

### Design Notes:
- Use color gradient: red (0%) to green (100%)
- Each bar should reach ~97-98% and be green
- Add small comparison text: "vs. human expert: 85-90%"

---

## SLIDE 18: CONFUSION MATRIX ✓

### Visual Layout:
```
┌────────────────────────────────────────┐
│  Test Set Performance: Confusion Matrix│
│                                        │
│         Predicted Healthy  Diseased   │
│  Actual Healthy    38         1       │
│         Diseased    3        57       │
│                                        │
│  Interpretation:                      │
│  • True Negatives: 38 ✓                │
│  • False Positives: 1 ⚠️               │
│  • False Negatives: 3 ⚠️               │
│  • True Positives: 57 ✓               │
│                                        │
│  Total Accuracy: 95/99 = 96%          │
│  (Different from 97.4% on full set)   │
│                                        │
└────────────────────────────────────────┘
```

### Image Specs:
- 2×2 matrix with color heatmap
- Correct predictions (diagonal): bright green
- Errors (off-diagonal): yellow/red
- Cell size proportional to count (57 and 38 larger than 1 and 3)
- Clear labels: Predicted vs. Actual

### Design Notes:
- Use strong color contrast (green for correct, red for wrong)
- Add small annotations: TP, FP, FN, TN labels
- Include interpretation text below

---

## SLIDE 19: PIPELINE IN ACTION ✓

### Visual Layout:
```
┌──────────────────────────────────────┐
│  Two-Stage Pipeline: Complete Flow   │
│                                      │
│  1️⃣ INPUT                           │
│     [Raw messy field image]          │
│     ↓                                │
│  2️⃣ SEGMENTATION                    │
│     [YOLOv8-seg processing]          │
│     ↓                                │
│  3️⃣ CROPPING                        │
│     [Isolated leaf grid]             │
│     ↓                                │
│  4️⃣ CLASSIFICATION                  │
│     [EfficientNet processing]        │
│     ↓                                │
│  5️⃣ OUTPUT                          │
│     [Annotated image with green &    │
│      red boxes]                      │
│                                      │
└──────────────────────────────────────┘
```

### Image Specs:
- 5-step vertical flow
- Each step shows actual image at that stage
- Step 1: Messy raw image
- Step 2: Segmentation masks
- Step 3: Grid of isolated leaves
- Step 4: Processing indicator
- Step 5: Final annotated output with green/red boxes

### Design Notes:
- Use large numbered circles (1️⃣-5️⃣) for steps
- Arrows between each step
- Each actual image ~250×250px
- Final output shows confidence scores (e.g., "Healthy, 98%")

---

## SLIDE 20: STREAMLIT APPLICATION ✓

### Visual Layout:
```
┌────────────────────────────────────────┐
│  Deployed System: Streamlit Dashboard  │
│                                        │
│  ┌──── Tomato Disease Detection ────┐ │
│  │                                  │ │
│  │  Upload Image or Video:          │ │
│  │  [Upload Button] 📁              │ │
│  │                                  │ │
│  │  ┌─── Results ─────┐            │ │
│  │  │ [Processed      │            │ │
│  │  │  Image]         │            │ │
│  │  │                 │            │ │
│  │  │ Healthy: 95%  │            │ │
│  │  │ Diseased: 5%  │            │ │
│  │  │                 │            │ │
│  │  │ [Download BTN] │            │ │
│  │  └─────────────────┘            │ │
│  │                                  │ │
│  └──────────────────────────────────┘ │
│                                        │
│  Key Features:                         │
│  ✓ Image & video upload               │
│  ✓ Real-time processing               │
│  ✓ Downloadable results               │
│  ✓ Simple UI (no ML jargon)           │
│                                        │
└────────────────────────────────────────┘
```

### Image Specs:
- Screenshot or mockup of Streamlit interface
- Clean, simple design
- Clear upload button
- Sample processed image shown
- Results displayed with percentage bars

### Design Notes:
- Keep interface design minimal and clean
- Use consistent green/red color scheme
- Show actual Streamlit elements (if possible: upload widget, sliders, buttons)

---

## SLIDE 21: IMAGE PROCESSING IN APP ✓

### Visual Layout:
```
┌──────────────────────────────────────┐
│  App Processing Pipeline              │
│                                      │
│  User Upload                         │
│    ↓                                 │
│  YOLOv8-seg Detection                │
│    ↓                                 │
│  Leaf Check (any detected?)          │
│    ├─ If Yes → Extract crops        │
│    └─ If No → Skip forward           │
│    ↓                                 │
│  Batch Processing                    │
│    (All leaves at once = faster)    │
│    ↓                                 │
│  EfficientNet Classification         │
│    ↓                                 │
│  Visualize Results                   │
│    (Green for healthy,               │
│     Red for diseased)                │
│    ↓                                 │
│  User Downloads Annotated Output     │
│                                      │
└──────────────────────────────────────┘
```

### Image Specs:
- Flowchart showing processing steps
- Each step represented with icon
- Decision point (diamond shape for "any leaves detected?")
- Visual emphasis on batch processing efficiency

### Design Notes:
- Use consistent arrow styling
- Color code: blue for input, gray for processing, green for output
- Add note about batch processing efficiency

---

## SLIDE 22: EDGE DEVICE DEPLOYMENT ✓

### Visual Layout:
```
┌────────────────────────────────────────┐
│  Future: Deployment on Edge Devices    │
│                                        │
│  [Raspberry Pi]  [Jetson Nano]       │
│   $30, 5-10W     $100, 10W            │
│   ✓✓✓            ✓✓✓✓                │
│                                        │
│  [Smartphone]    [Drone]              │
│   Android app    Real-time mapping    │
│   ✓✓✓✓           ✓✓✓✓               │
│                                        │
│  Benefits:                             │
│  • Offline operation (no internet)    │
│  • Real-time field deployment         │
│  • Affordable (₹2,500 for RPi)       │
│  • Farmer-controlled                  │
│                                        │
└────────────────────────────────────────┘
```

### Image Specs:
- 2×2 grid of device photos
- Raspberry Pi (actual product photo)
- NVIDIA Jetson Nano (actual product)
- Smartphone showing app interface
- Drone with mounted camera
- Cost and power consumption labeled

### Design Notes:
- Use actual product photos (high quality)
- Add star ratings or checkmarks for capability comparison
- Highlight affordability and accessibility

---

## SLIDE 23: MULTI-DISEASE EXPANSION ✓

### Visual Layout:
```
┌────────────────────────────────────────┐
│  Future: Multi-Disease Detection       │
│                                        │
│  Current (Binary):                     │
│  Healthy ├─ Early Blight              │
│                                        │
│  Future (Multi-class):                 │
│  Healthy ├─ Early Blight              │
│         ├─ Late Blight                │
│         ├─ Septoria Leaf Spot         │
│         ├─ Fusarium Wilt              │
│         └─ Bacterial Spot             │
│                                        │
│  [Images of each disease type]         │
│                                        │
│  Expansion Plan:                       │
│  ✓ Stage 1 (Segmentation): Unchanged  │
│  ✓ Stage 2 (Classifier): Retrain      │
│  ✓ 2 classes → 5-10 classes           │
│  ✓ Modular architecture = easy scale  │
│                                        │
└────────────────────────────────────────┘
```

### Image Specs:
- Left: Current binary classification (2 classes)
- Right: Future multi-class (6+ classes)
- Small thumbnail images for each disease type
- Visual hierarchy showing class structure

### Design Notes:
- Use tree/branching diagram to show class hierarchy
- Each disease shown with distinctive leaf image
- Highlight that Stage 1 doesn't change

---

## SLIDE 24: GEO-SPATIAL MAPPING ✓

### Visual Layout:
```
┌────────────────────────────────────────┐
│  Future: Precision Ag with Drone       │
│                                        │
│  [Drone Photo of Field]                │
│    ↓                                   │
│  Real-time disease scanning            │
│    ↓                                   │
│  GPS-tagged results                    │
│    ↓                                   │
│  ┌──── Disease Heatmap ────┐          │
│  │ 🟢 Green: Healthy       │          │
│  │ 🟡 Yellow: Early stage  │          │
│  │ 🔴 Red: Severe          │          │
│  └─────────────────────────┘          │
│                                        │
│  Benefits:                             │
│  • Targeted pesticide application     │
│  • Reduce chemical use by 80%+        │
│  • Better yields, healthier crops     │
│  • Environmental + economic benefit   │
│                                        │
└────────────────────────────────────────┘
```

### Image Specs:
- Top: Drone flying over tomato field (actual or realistic rendering)
- Bottom: Digital heatmap showing disease distribution
- Green zones (healthy), Yellow zones (early disease), Red zones (severe)
- GPS grid overlaid on map

### Design Notes:
- Use realistic drone perspective
- Heatmap should be colorful and informative
- Show scale/legend clearly

---

## SLIDE 25: CONCLUSION ✓

### Visual Layout:
```
┌────────────────────────────────────────┐
│  Conclusion: What We Achieved          │
│                                        │
│        ✅ Two-stage pipeline           │
│        ✅ 97.4% accuracy              │
│        ✅ Real-world robustness       │
│        ✅ Deployment pathway          │
│        ✅ Open-source ready           │
│                                        │
│  Impact:                               │
│  🌍 Democratizes disease detection    │
│  👨‍🌾 Accessible to small-scale farmers  │
│  🌱 Supports sustainable agriculture  │
│  💚 Reduces unnecessary pesticides    │
│                                        │
│  Thank you for your attention!         │
│                                        │
└────────────────────────────────────────┘
```

### Image Specs:
- Central trophy or achievement icon
- Checkmarks around it (achievements)
- Icons representing impact:
  - Globe icon (democratization)
  - Farmer icon (accessibility)
  - Plant icon (sustainability)
  - Leaf icon (environment)

### Design Notes:
- Use bold, inspiring colors
- Large icons (100+px each)
- Positive messaging throughout
- Celebratory tone

---

## SLIDE 26: THANK YOU & Q&A ✓

### Visual Layout:
```
┌────────────────────────────────────────┐
│                                        │
│                                        │
│        Thank You!                      │
│                                        │
│  [Beautiful field image               │
│   or tomato plant image]               │
│                                        │
│  Questions?                            │
│                                        │
│                                        │
│  Contact: [email or institution]      │
│  Team: Dhruv, Himanshu, Dheeraj,     │
│         Neharika                       │
│  MNNIT Allahabad                      │
│                                        │
└────────────────────────────────────────┘
```

### Image Specs:
- Full-screen or near-full beautiful field image
- Healthy, vibrant tomato plants
- Warm, inspiring lighting
- High quality, professional photography

### Text Formatting:
- "Thank You": Large (60+pt), centered
- "Questions?": 48pt
- Contact info: 24pt, subtle color

### Design Notes:
- Simple and inspiring
- Let the beautiful field image speak
- Minimal text, maximum impact
- Warm color palette

---

# FINAL CHECKLIST

Before Presentation Day:

- [ ] All slides have high-quality images (no pixelated/low-res)
- [ ] Text is readable from 10 feet away (test on projector)
- [ ] Color scheme is consistent (green=healthy, orange/red=diseased)
- [ ] Transitions between slides are smooth
- [ ] Presenter notes are clear and concise
- [ ] Team has practiced handoffs 3+ times
- [ ] Backup slides are ready (in case of technical questions)
- [ ] Presentation is under 20 minutes total
- [ ] One practice run with projector if possible

---