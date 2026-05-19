# CSC 110 — Introductory Computer Science Lab
## Lab Exercise: Object Detection with YOLO
**VUA CSC 110 | First-Year Students**

---

## 🎯 What You Will Learn

By the end of this lab, you will be able to:
- Understand what a computer vision model is
- Install and use the YOLO (You Only Look Once) model
- Write a Python script that detects objects in an image
- Read and understand the detection results

**Estimated Time:** 45 – 60 minutes

---

## 📖 Background: What is YOLO?

**YOLO** stands for **You Only Look Once**. It is a popular AI model that can look at a picture and identify objects in it — like people, cars, dogs, chairs, bottles, and many more.

Think of it like this: if you showed a photo to a friend and asked *"what do you see?"*, they might say *"I see a dog and a red chair."* YOLO does exactly that — but with a computer!

> **Fun fact:** YOLO can detect up to **80 different types of objects** in a single image!

---

## 🛠️ Part 1: Setting Up Google Colab

For this lab, we will use **Google Colab** — a free online tool from Google that lets you write and run Python code directly in your web browser. You do **not** need to install anything on your computer!

### Step 1 — Open Google Colab

1. Open **Google Chrome** (recommended) on your computer or phone
2. Go to: **[https://colab.research.google.com](https://colab.research.google.com)**
3. Sign in with your **Google / Gmail account**

---

### Step 2 — Create a New Notebook

1. Click **File → New notebook** from the top menu
2. A new page opens with an empty box — this is called a **cell**, and it is where you type your code
3. Click on **"Untitled0"** at the very top and rename it to:
   ```
   CSC110_YOLO_Lab
   ```

> 💡 **What is a cell?** Colab notebooks are made up of **cells**. Each cell holds a piece of code. You run a cell by clicking the **▶ play button** on its left, or pressing **Shift + Enter** on your keyboard. The output appears directly below the cell.

---

### Step 3 — Get a Test Image

You need a photo for YOLO to analyse. Choose **any photo** you like — a street scene, a photo from your phone, or anything with people, animals, or objects in it.

Keep it ready on your device. You will upload it to Colab in Part 2.

---

## 💻 Part 2: Writing Your YOLO Code in Colab

You will write your code across **4 cells**. Add a new cell each time by clicking **"+ Code"** at the top-left of the page.

---

### ▶ Cell 1 — Install YOLO

In your **first cell**, type the following and run it:

```python
# Cell 1: Install the ultralytics library (this gives us YOLO)
!pip install ultralytics -q
```

> ⚠️ Wait for it to finish before moving on. You will see a green ✅ when it is done. This may take up to 30 seconds.

---

### ▶ Cell 2 — Upload Your Image

Add a **new cell** and type:

```python
# Cell 2: Upload an image from your computer into Colab
from google.colab import files

uploaded = files.upload()  # A file picker button will appear below — click it to upload your photo
```

Run this cell. A **"Choose Files"** button will appear — click it and select your photo from your device. Once uploaded, you will see the filename printed below.

> 💡 Note the exact filename (e.g., `my_photo.jpg`) — you will need it in the next step.

---

### Step 5 — Type the Detection Code

Add a **new cell** and type the code below. Read each comment (lines starting with `#`) — they explain what each line does.

```python
# ============================================================
# Cell 3 — CSC 110 Lab: Object Detection with YOLO
# ============================================================

# Step A: Import the YOLO class from the ultralytics library
from ultralytics import YOLO

# Step B: Load the YOLO model
# "yolov8n.pt" is the smallest (and fastest) version of YOLO
# The "n" stands for "nano" - great for beginners!
# The first time you run this, it will download the model automatically.
model = YOLO("yolov8n.pt")

# Step C: Run object detection on your image
# ⚠️ Replace "your_photo.jpg" with the exact filename you uploaded in Cell 2
results = model("your_photo.jpg")

# Step D: Print the results in a readable way
print("\n========== DETECTION RESULTS ==========")

# Loop through each result (there is usually just one for a single image)
for result in results:

    # Get the list of detected objects
    boxes = result.boxes

    # Check if anything was detected
    if len(boxes) == 0:
        print("No objects were detected in the image.")
    else:
        print(f"Number of objects detected: {len(boxes)}\n")

        # Loop through each detected object
        for i, box in enumerate(boxes):

            # Get the class ID (a number that represents the object type)
            class_id = int(box.cls)

            # Get the name of the detected object (e.g., "person", "car")
            object_name = result.names[class_id]

            # Get the confidence score (how sure the model is, from 0 to 1)
            confidence = float(box.conf)

            # Convert confidence to a percentage
            confidence_percent = confidence * 100

            # Print the result for this object
            print(f"Object {i+1}: {object_name}")
            print(f"   Confidence: {confidence_percent:.1f}%")
            print()

print("========================================")
print("Lab complete! Great job!")
```

---

### Step 6 — Run the Cell

Click the **▶ play button** on the left of Cell 3, or press **Shift + Enter**.

> 💡 The first time you run this, YOLO will download a small model file (~6 MB). You will see a progress bar — this is normal! After that, the detection runs instantly.

---

## 📊 Part 3: Understanding the Output

When your script runs successfully, you should see something like this in the terminal:

```
========== DETECTION RESULTS ==========
Number of objects detected: 3

Object 1: person
   Confidence: 91.3%

Object 2: car
   Confidence: 87.6%

Object 3: backpack
   Confidence: 74.2%

========================================
Lab complete! Great job!
```

### What does "Confidence" mean?
The confidence score tells you **how sure** the model is about its detection:
- **90% and above** → Very confident. The model is quite sure.
- **50% – 89%** → Fairly confident. Probably correct.
- **Below 50%** → Not very sure. The result might be wrong.

---

## 🖼️ Bonus: View the Annotated Image in Colab (Optional)

Want to see the image with coloured boxes drawn around each detected object? Add a **new cell (Cell 4)** and type:

```python
# Cell 4 (Bonus): Save and display the annotated image inside Colab
from IPython.display import Image as IPImage

# Save the result image with boxes drawn on it
results[0].save(filename="output_image.jpg")

# Display it directly inside the notebook
IPImage("output_image.jpg")
```

Run the cell — the annotated image will appear **right inside your Colab notebook**!

---

## ✅ Part 4: Lab Tasks

Complete all the tasks below and show your results to your instructor.

| # | Task | Done? |
|---|------|-------|
| 1 | Open Google Colab and create your notebook | ☐ |
| 2 | Run Cell 1 — install ultralytics successfully | ☐ |
| 3 | Run Cell 2 — upload your image | ☐ |
| 4 | Run Cell 3 — detect objects in your image | ☐ |
| 5 | Record: how many objects were detected? | ☐ |
| 6 | Upload a **second different image** and run detection again | ☐ |
| 7 | Run Cell 4 — view the annotated image (Bonus) | ☐ |

---

## 📝 Reflection Questions

Answer these questions in your notebook or on a sheet of paper:

1. What was the **most surprising** object that YOLO detected in your image?
2. Was there any object in your image that YOLO **missed** or got **wrong**? Why do you think that happened?
3. In your own words, what do you think a **confidence score** means?
4. Can you think of **two real-world applications** where object detection like YOLO would be useful?

---

## 📤 Part 5: Submitting Your Work to GitHub

You will submit your lab by uploading your Colab notebook file to the course GitHub repository. You do **not** need to install Git or know any Git commands — everything is done by clicking in your browser.

---

### Step 1 — Download Your Notebook from Colab

1. In your Colab notebook, click **File** in the top menu
2. Click **Download → Download .ipynb**
3. A file ending in `.ipynb` will be saved to your computer (e.g., `CSC110_YOLO_Lab.ipynb`)

> 💡 `.ipynb` is the file format for Colab/Jupyter notebooks. It contains all your code and output.

---

### Step 2 — Open the Course GitHub Repository

1. Open your browser and go to the course repository link provided by your instructor:
   ```
   https://github.com/[instructor-username]/[repo-name]
   ```
2. Sign in to your **GitHub account** (create a free one at https://github.com if you don't have one)

---

### Step 3 — Navigate to Your Submission Folder

1. Inside the repository, look for a folder called **`submissions/`** or a folder with your name — your instructor will tell you which one to use
2. Click on that folder to open it

---

### Step 4 — Upload Your Notebook File

1. Inside the folder, click the **"Add file"** button (top-right area)
2. Select **"Upload files"** from the dropdown

   ![Upload files option]

3. Either **drag and drop** your `.ipynb` file onto the page, or click **"choose your files"** and select it from your computer
4. You will see your file appear in the upload area

---

### Step 5 — Commit (Save) Your File

Scroll down the page. You will see a section called **"Commit changes"**.

1. In the first text box, type a short message describing what you are submitting, for example:
   ```
   CSC110 YOLO Lab submission - [Your Full Name]
   ```
2. Leave everything else as it is
3. Click the green **"Commit changes"** button

✅ Your file is now submitted! You can revisit the folder on GitHub anytime to confirm it is there.

---

### ⚠️ Important Notes

- Make sure you **run all cells** in your notebook before downloading, so your output is saved inside the file
- Name your file clearly before uploading, e.g.: `YOLO_Lab_YourFirstName_YourLastName.ipynb`
- If you cannot find the submissions folder or get an error saying you don't have permission, contact your instructor — they may need to add you as a contributor to the repository

---

## 🔧 Troubleshooting

| Problem | Solution |
|--------|----------|
| `ModuleNotFoundError: No module named 'ultralytics'` | Re-run Cell 1 to install the library again |
| `FileNotFoundError` for the image | Make sure the filename in Cell 3 **exactly matches** what was uploaded in Cell 2 (spelling, uppercase/lowercase, file extension) |
| Upload button does not appear | Re-run Cell 2 and wait a moment for the button to load |
| Download seems stuck | Be patient — the model file is ~6 MB. Check your internet connection |
| Session disconnected / runtime reset | Re-run all cells from Cell 1 in order — Colab resets if left idle |
| Image not displaying (Bonus cell) | Make sure Cell 3 ran successfully before running Cell 4 |
| GitHub says "permission denied" on upload | Ask your instructor to add your GitHub username as a contributor |
| Cannot find the submissions folder | Check with your instructor for the exact folder name and repo link |

---

## 📚 What We Used Today

| Term | Meaning |
|------|---------|
| **Google Colab** | A free online tool for writing and running Python code in your browser |
| **Cell** | A box in a Colab notebook where you type and run code |
| **YOLO** | A fast AI model for detecting objects in images |
| **Model** | A trained AI program that can make predictions |
| **Confidence score** | How sure the AI is about its answer (0–100%) |
| **Bounding box** | A rectangle drawn around a detected object |
| **ultralytics** | A Python library that provides easy access to YOLO |
| **.ipynb** | The file format for Colab/Jupyter notebooks |
| **GitHub** | A website for storing and sharing code files |
| **Commit** | Saving a file to a GitHub repository |

---

*CSC 110 — Introduction to Computer Science | VUA*
*Lab prepared for first-year students*
