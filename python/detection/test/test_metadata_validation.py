import numpy as np

from detection.data_pipeline import (
    prepare_dataset_data,
    build_inference_dataset,
    build_inference_metadata,
    load_image,
)

from dataset.dataset_utils import (
    get_bounding_boxes_train_pairs,
    train_validation_split,
)

from detection.inference import (
    create_combined_views,
)

from detection.detection_config import (
    SPLIT_RATIO,
    SEED,
)


def test_inference_dataset_alignment(image_paths, image_shapes, num_images=3):
    image_paths = image_paths[:num_images]
    image_shapes = image_shapes[:num_images]

    # Dataset prodotto dalla nuova pipeline
    inference_ds = build_inference_dataset(image_paths)

    # Metadata corrispondenti
    metadata = build_inference_metadata(image_shapes)

    # Estrazione delle views dal dataset batchato
    dataset_views = []

    for batch in inference_ds:
        for view in batch.numpy():
            dataset_views.append(view)

    # Views di riferimento generate direttamente
    expected_views = []

    for image_path in image_paths:
        image = load_image(image_path).numpy()

        views, _ = create_combined_views(image)

        for view in views:
            expected_views.append(
                view.astype(np.float32)
            )

    print("\n===================================")
    print("INFERENCE DATASET ALIGNMENT TEST")
    print("===================================")

    print("Images tested:", num_images)
    print("Expected views:", len(expected_views))
    print("Dataset views:", len(dataset_views))
    print("Metadata:", len(metadata))

    assert len(dataset_views) == len(expected_views), (
        f"Wrong number of dataset views: "
        f"{len(dataset_views)} != {len(expected_views)}"
    )

    assert len(metadata) == len(expected_views), (
        f"Wrong number of metadata entries: "
        f"{len(metadata)} != {len(expected_views)}"
    )

    for i, (dataset_view, expected_view) in enumerate(
        zip(dataset_views, expected_views)
    ):
        assert dataset_view.shape == expected_view.shape, (
            f"View {i}: shape mismatch: "
            f"{dataset_view.shape} != {expected_view.shape}"
        )

        max_error = np.max(
            np.abs(
                dataset_view.astype(np.float32)
                - expected_view.astype(np.float32)
            )
        )

        if max_error > 0:
            raise AssertionError(
                f"View {i} does not match. "
                f"Maximum error: {max_error}"
            )

        print(
            f"[{i + 1}/{len(expected_views)}] "
            f"image={metadata[i]['image_index']} | "
            f"local={metadata[i]['local_view_index']} | "
            f"{'PATCH' if metadata[i]['is_patch'] else 'FULL'} | "
            f"OK"
        )

    print("\nTEST PASSED")
    print("Dataset views and metadata are perfectly aligned.")


def main():
    # Stesso insieme di immagini utilizzato dalla pipeline normale
    pairs = get_bounding_boxes_train_pairs()

    # Ricostruiamo esattamente lo stesso train/validation split
    _, val_pairs = train_validation_split(
        pairs,
        SPLIT_RATIO,
        SEED
    )

    print("Validation images:", len(val_pairs))

    # Ci interessano path e dimensioni originali.
    # Classes e bbox non servono per questo test.
    (
        val_image_paths,
        _,
        _,
        val_image_shapes,
    ) = prepare_dataset_data(val_pairs)

    test_inference_dataset_alignment(val_image_paths, val_image_shapes, num_images=3,)


if __name__ == "__main__":
    main()