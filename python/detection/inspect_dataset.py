from pathlib import Path
import json
from collections import Counter

# ============================================================
# DATASET OVERVIEW
# ============================================================
# This section counts and analyzes annotations, objects, classes,
# geometry types, and tags in the bounding box dataset.

DATASET_DIR = Path("dataset") / "fsoco_bounding_boxes_train"

class_counter = Counter()
geometry_counter = Counter()
tag_counter = Counter()

num_annotations = 0
num_objects = 0

# Iterate through all annotation JSON files and collect statistics
for json_path in DATASET_DIR.glob("*/ann/*.json"):
    num_annotations += 1

    with open(json_path, "r", encoding="utf-8") as f:
        annotation = json.load(f)

    for obj in annotation.get("objects", []):
        num_objects += 1

        class_name = obj["classTitle"]
        geometry_type = obj["geometryType"]

        class_counter[class_name] += 1
        geometry_counter[geometry_type] += 1

        for tag in obj.get("tags", []):
            tag_counter[tag["name"]] += 1


print("Annotations:", num_annotations)
print("Objects:", num_objects)

print("\nClasses:")
for name, count in class_counter.items():
    print(f"{name}: {count}")

print("\nGeometry types:")
for name, count in geometry_counter.items():
    print(f"{name}: {count}")

print("\nTags:")
for name, count in tag_counter.items():
    print(f"{name}: {count}")

print("\nPercentage of each category:")
for name, count in class_counter.items():
    print(f"{name}: {(count/num_objects):.2f}")

# ============================================================
# BOUNDING BOX ANALYSIS
# ============================================================
# This section analyzes bounding box dimensions and characteristics
# including width, height, area, and distribution by object class

from collections import defaultdict
from statistics import mean, median

# Initialize lists to collect global statistics on bounding boxes
bbox_widths = []
bbox_heights = []
bbox_areas = []
relative_areas = []

# Initialize separate statistics collectors for each class
# This allows us to analyze bounding box distribution per class
bbox_widths_by_class = defaultdict(list)
bbox_heights_by_class = defaultdict(list)
bbox_areas_by_class = defaultdict(list)
relative_areas_by_class = defaultdict(list)

# Track size categories to understand small/medium/large object distribution
size_counter = Counter()
size_counter_by_class = defaultdict(Counter)


# Process all annotation files and extract bounding box information
for json_path in DATASET_DIR.glob("*/ann/*.json"):

    with open(json_path, "r", encoding="utf-8") as f:
        annotation = json.load(f)

    # Get image dimensions for calculating relative areas
    image_width = annotation["size"]["width"]
    image_height = annotation["size"]["height"]

    image_area = image_width * image_height

    for obj in annotation.get("objects", []):

        # Skip non-rectangle geometries (this dataset should only contain rectangles)
        if obj["geometryType"] != "rectangle":
            continue

        class_name = obj["classTitle"]

        # Extract bounding box corner coordinates
        (x_min, y_min), (x_max, y_max) = obj["points"]["exterior"]

        # Calculate bounding box dimensions
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        bbox_area = bbox_width * bbox_height

        # Calculate area relative to image (0-1 scale)
        relative_area = bbox_area / image_area

        # Collect global statistics
        bbox_widths.append(bbox_width)
        bbox_heights.append(bbox_height)
        bbox_areas.append(bbox_area)
        relative_areas.append(relative_area)

        # Collect per-class statistics for detailed analysis
        bbox_widths_by_class[class_name].append(bbox_width)
        bbox_heights_by_class[class_name].append(bbox_height)
        bbox_areas_by_class[class_name].append(bbox_area)
        relative_areas_by_class[class_name].append(relative_area)

        # Categorize bounding box size
        # Used to understand distribution of small vs large objects
        max_dimension = max(bbox_width, bbox_height)

        if max_dimension < 16:
            size_category = "<16 px"

        elif max_dimension < 32:
            size_category = "16-32 px"

        elif max_dimension < 64:
            size_category = "32-64 px"

        elif max_dimension < 128:
            size_category = "64-128 px"

        else:
            size_category = ">=128 px"

        size_counter[size_category] += 1
        size_counter_by_class[class_name][size_category] += 1


# ============================================================
# GLOBAL BOUNDING BOX STATISTICS
# ============================================================
# Display comprehensive statistics about bounding box dimensions
# across the entire dataset

print("\n===== BOUNDING BOX STATISTICS =====")

print("\nWidth:")
print(f"Minimum: {min(bbox_widths)} px")
print(f"Maximum: {max(bbox_widths)} px")
print(f"Mean: {mean(bbox_widths):.2f} px")
print(f"Median: {median(bbox_widths):.2f} px")

print("\nHeight:")
print(f"Minimum: {min(bbox_heights)} px")
print(f"Maximum: {max(bbox_heights)} px")
print(f"Mean: {mean(bbox_heights):.2f} px")
print(f"Median: {median(bbox_heights):.2f} px")

print("\nArea:")
print(f"Minimum: {min(bbox_areas)} px²")
print(f"Maximum: {max(bbox_areas)} px²")
print(f"Mean: {mean(bbox_areas):.2f} px²")
print(f"Median: {median(bbox_areas):.2f} px²")

print("\nRelative area:")
print(f"Minimum: {min(relative_areas) * 100:.6f}%")
print(f"Maximum: {max(relative_areas) * 100:.6f}%")
print(f"Mean: {mean(relative_areas) * 100:.6f}%")
print(f"Median: {median(relative_areas) * 100:.6f}%")


# ============================================================
# PER-CLASS BOUNDING BOX STATISTICS
# ============================================================
# Display detailed statistics broken down by object class
# to understand class-specific object characteristics

print("\n===== BOUNDING BOX STATISTICS BY CLASS =====")

for class_name in bbox_areas_by_class:

    widths = bbox_widths_by_class[class_name]
    heights = bbox_heights_by_class[class_name]
    areas = bbox_areas_by_class[class_name]
    rel_areas = relative_areas_by_class[class_name]

    print(f"\n--- {class_name} ---")

    print(
        f"Width: "
        f"mean = {mean(widths):.2f}, "
        f"median = {median(widths):.2f}, "
        f"min = {min(widths)}, "
        f"max = {max(widths)}"
    )

    print(
        f"Height: "
        f"mean = {mean(heights):.2f}, "
        f"median = {median(heights):.2f}, "
        f"min = {min(heights)}, "
        f"max = {max(heights)}"
    )

    print(
        f"Area: "
        f"mean = {mean(areas):.2f}, "
        f"median = {median(areas):.2f}, "
        f"min = {min(areas)}, "
        f"max = {max(areas)}"
    )

    print(
        f"Relative area: "
        f"mean = {mean(rel_areas) * 100:.6f}%, "
        f"median = {median(rel_areas) * 100:.6f}%"
    )


# ============================================================
# DISTRIBUZIONE DELLE DIMENSIONI
# ============================================================

print("\n===== OBJECT SIZE DISTRIBUTION =====")

size_order = [
    "<16 px",
    "16-32 px",
    "32-64 px",
    "64-128 px",
    ">=128 px"
]

for size in size_order:

    count = size_counter[size]
    percentage = (count / num_objects) * 100

    print(
        f"{size}: "
        f"{count} objects "
        f"({percentage:.2f}%)"
    )


# ============================================================
# DISTRIBUZIONE DELLE DIMENSIONI PER CLASSE
# ============================================================

print("\n===== OBJECT SIZE DISTRIBUTION BY CLASS =====")

for class_name in class_counter:

    print(f"\n--- {class_name} ---")

    total_class_objects = class_counter[class_name]

    for size in size_order:

        count = size_counter_by_class[class_name][size]

        percentage = (
            count / total_class_objects * 100
            if total_class_objects > 0
            else 0
        )

        print(
            f"{size}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )

# ============================================================
# STEP 3 - IMAGE COMPOSITION ANALYSIS
# ============================================================

from collections import defaultdict
from itertools import combinations
from statistics import mean, median


TARGET_CLASSES = [
    "blue_cone",
    "yellow_cone",
    "orange_cone",
    "large_orange_cone"
]

# Numero totale di coni per immagine
objects_per_image = []

# Numero di immagini in cui compare ciascuna classe
images_with_class = Counter()

# Numero di oggetti di ogni classe, immagine per immagine
class_counts_per_image = defaultdict(list)

# Numero di classi diverse presenti in ogni immagine
num_classes_per_image = Counter()

# Co-occorrenza tra classi
class_cooccurrence = Counter()

# Tag suddivisi per classe
tags_by_class = defaultdict(Counter)

# Numero di immagini che contengono unknown_cone
images_with_unknown = 0


for json_path in DATASET_DIR.glob("*/ann/*.json"):

    with open(json_path, "r", encoding="utf-8") as f:
        annotation = json.load(f)

    objects = annotation.get("objects", [])

    # --------------------------------------------------------
    # Numero totale di coni nell'immagine
    # --------------------------------------------------------

    objects_per_image.append(len(objects))

    # Conta le classi presenti nella singola immagine
    image_class_counter = Counter()

    for obj in objects:

        class_name = obj["classTitle"]

        image_class_counter[class_name] += 1

        # Tag dell'oggetto suddivisi per classe
        for tag in obj.get("tags", []):
            tags_by_class[class_name][tag["name"]] += 1

    # --------------------------------------------------------
    # Immagini contenenti ciascuna classe
    # --------------------------------------------------------

    for class_name in image_class_counter:
        images_with_class[class_name] += 1

    if "unknown_cone" in image_class_counter:
        images_with_unknown += 1

    # --------------------------------------------------------
    # Numero di oggetti di ogni classe nell'immagine
    # --------------------------------------------------------

    for class_name in class_counter:
        class_counts_per_image[class_name].append(
            image_class_counter[class_name]
        )

    # --------------------------------------------------------
    # Numero di CLASSI TARGET presenti nell'immagine
    # unknown_cone non viene considerato come classe Race UP
    # --------------------------------------------------------

    target_classes_present = [
        class_name
        for class_name in TARGET_CLASSES
        if image_class_counter[class_name] > 0
    ]

    num_classes_per_image[len(target_classes_present)] += 1

    # --------------------------------------------------------
    # Co-occorrenza delle classi target
    # --------------------------------------------------------

    for class1, class2 in combinations(
        sorted(target_classes_present), 2
    ):
        class_cooccurrence[(class1, class2)] += 1


# ============================================================
# 3.1 - NUMBER OF CONES PER IMAGE
# ============================================================

print("\n===== STEP 3: IMAGE COMPOSITION =====")

print("\n===== CONES PER IMAGE =====")

print(f"Minimum: {min(objects_per_image)}")
print(f"Maximum: {max(objects_per_image)}")
print(f"Mean: {mean(objects_per_image):.2f}")
print(f"Median: {median(objects_per_image):.2f}")


# ============================================================
# 3.2 - DISTRIBUTION OF CONES PER IMAGE
# ============================================================

cones_per_image_ranges = Counter()

for count in objects_per_image:

    if count == 0:
        category = "0"

    elif count <= 5:
        category = "1-5"

    elif count <= 10:
        category = "6-10"

    elif count <= 20:
        category = "11-20"

    elif count <= 40:
        category = "21-40"

    else:
        category = ">40"

    cones_per_image_ranges[category] += 1


print("\n===== CONES PER IMAGE DISTRIBUTION =====")

range_order = [
    "0",
    "1-5",
    "6-10",
    "11-20",
    "21-40",
    ">40"
]

for category in range_order:

    count = cones_per_image_ranges[category]

    percentage = (
        count / num_annotations * 100
        if num_annotations > 0
        else 0
    )

    print(
        f"{category}: "
        f"{count} images "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 3.3 - IMAGES CONTAINING EACH CLASS
# ============================================================

print("\n===== IMAGES CONTAINING EACH CLASS =====")

for class_name in class_counter:

    image_count = images_with_class[class_name]

    percentage = image_count / num_annotations * 100

    print(
        f"{class_name}: "
        f"{image_count} images "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 3.4 - NUMBER OF CONES OF EACH CLASS WHEN PRESENT
# ============================================================

print("\n===== CONES OF EACH CLASS WHEN PRESENT =====")

for class_name in class_counter:

    # Escludiamo gli zero:
    # vogliamo sapere quanti coni ci sono mediamente
    # QUANDO la classe è presente nell'immagine
    positive_counts = [
        count
        for count in class_counts_per_image[class_name]
        if count > 0
    ]

    if not positive_counts:
        continue

    print(f"\n--- {class_name} ---")

    print(
        f"Images: {len(positive_counts)}"
    )

    print(
        f"Mean cones when present: "
        f"{mean(positive_counts):.2f}"
    )

    print(
        f"Median cones when present: "
        f"{median(positive_counts):.2f}"
    )

    print(
        f"Maximum cones in one image: "
        f"{max(positive_counts)}"
    )


# ============================================================
# 3.5 - NUMBER OF DIFFERENT TARGET CLASSES PER IMAGE
# ============================================================

print("\n===== NUMBER OF DIFFERENT TARGET CLASSES PER IMAGE =====")

for number_of_classes in range(5):

    count = num_classes_per_image[number_of_classes]

    percentage = count / num_annotations * 100

    print(
        f"{number_of_classes} classes: "
        f"{count} images "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 3.6 - CLASS CO-OCCURRENCE
# ============================================================

print("\n===== CLASS CO-OCCURRENCE =====")

for class1, class2 in combinations(
    sorted(TARGET_CLASSES), 2
):

    count = class_cooccurrence[(class1, class2)]

    percentage = count / num_annotations * 100

    print(
        f"{class1} + {class2}: "
        f"{count} images "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 3.7 - UNKNOWN CONES
# ============================================================

print("\n===== UNKNOWN CONES =====")

print(
    f"Images containing at least one unknown cone: "
    f"{images_with_unknown} "
    f"({images_with_unknown / num_annotations * 100:.2f}%)"
)


# ============================================================
# 3.8 - TAG DISTRIBUTION BY CLASS
# ============================================================

print("\n===== TAG DISTRIBUTION BY CLASS =====")

for class_name in class_counter:

    print(f"\n--- {class_name} ---")

    total_class_objects = class_counter[class_name]

    if not tags_by_class[class_name]:
        print("No tags")
        continue

    for tag_name, count in tags_by_class[class_name].items():

        percentage = count / total_class_objects * 100

        print(
            f"{tag_name}: "
            f"{count} "
            f"({percentage:.2f}% of {class_name})"
        )