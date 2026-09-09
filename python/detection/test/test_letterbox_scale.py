import numpy as np

from detection.inference import resize_with_letterbox


def test_letterbox(image_height, image_width, bbox):
    image = np.zeros(
        (image_height, image_width, 3),
        dtype=np.uint8
    )

    resized_image, scale_x, scale_y, pad_x, pad_y = resize_with_letterbox(
        image,
        (800, 800)
    )

    scale_x = float(scale_x.numpy())
    scale_y = float(scale_y.numpy())
    pad_x = int(pad_x.numpy())
    pad_y = int(pad_y.numpy())

    x_min, y_min, x_max, y_max = bbox

    # Forward:
    # original image -> letterboxed image
    resized_x_min = x_min * scale_x + pad_x
    resized_y_min = y_min * scale_y + pad_y
    resized_x_max = x_max * scale_x + pad_x
    resized_y_max = y_max * scale_y + pad_y

    # Inverse:
    # letterboxed image -> original image
    recovered_x_min = (resized_x_min - pad_x) / scale_x
    recovered_y_min = (resized_y_min - pad_y) / scale_y
    recovered_x_max = (resized_x_max - pad_x) / scale_x
    recovered_y_max = (resized_y_max - pad_y) / scale_y

    recovered_bbox = [recovered_x_min,recovered_y_min,recovered_x_max,recovered_y_max,
    ]

    print()
    print("===================================")
    print("IMAGE:", image_width, "x", image_height)
    print("===================================")

    print("Resized image shape:", resized_image.shape)

    print()
    print("scale_x:", scale_x)
    print("scale_y:", scale_y)
    print("pad_x:", pad_x)
    print("pad_y:", pad_y)

    print()
    print("Original bbox:")
    print(bbox)

    print()
    print("BBox in 800x800:")
    print([resized_x_min,resized_y_min,resized_x_max,resized_y_max,
    ])

    print()
    print("Recovered bbox:")
    print(recovered_bbox)

    errors = []

    for original, recovered in zip(bbox, recovered_bbox):
        errors.append(abs(original - recovered))

    max_error = max(errors)

    print()
    print("Errors:", errors)
    print("Maximum error:", max_error)

    if max_error < 0.001:
        print("TEST PASSED")
    else:
        print("TEST FAILED")


def main():

    test_letterbox(image_height=1480,image_width=2200,bbox=[500, 300, 800, 700]
    )

    test_letterbox(image_height=2200,image_width=1480,bbox=[200, 500, 900, 1500]
    )

    test_letterbox(image_height=800,image_width=800,bbox=[100, 100, 500, 600]
    )

    test_letterbox(image_height=1080,image_width=1920,bbox=[0, 0, 1920, 1080]
    )


if __name__ == "__main__":
    main()