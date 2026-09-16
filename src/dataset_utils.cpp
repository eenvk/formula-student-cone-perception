//Renzi

#include "dataset_utils.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <sstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <zlib.h>

namespace {

// Checks that image dimensions can be safely used to allocate masks.
void validate_image_size(int image_height, int image_width){
    if (image_height <= 0 || image_width <= 0) {
        throw std::invalid_argument("Image dimensions must be positive.");
    }
}

// Validates the basic format expected for semantic masks.
void validate_mask(const cv::Mat& mask){
    if (mask.empty() || mask.dims != 2 || mask.channels() != 1) {
        throw std::invalid_argument("Mask must be a non-empty single-channel image.");
    }
}

// Returns true for the image formats accepted by the dataset loader.
bool is_supported_image(const std::filesystem::path& path){
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
    return extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp";
}

// Converts fsoco class labels into the internal numeric identifiers.
int class_name_to_id(const std::string& class_name){
    static const std::unordered_map<std::string, int> class_ids = {
        {"yellow_cone", YELLOW_CONE_ID},
        {"blue_cone", BLUE_CONE_ID},
        {"orange_cone", SMALL_ORANGE_CONE_ID},
        {"small_orange_cone", SMALL_ORANGE_CONE_ID},
        {"large_orange_cone", BIG_ORANGE_CONE_ID},
        {"big_orange_cone", BIG_ORANGE_CONE_ID},
        {"unknown_cone", IGNORE_ID},
        {"seg_yellow_cone", YELLOW_CONE_ID},
        {"seg_blue_cone", BLUE_CONE_ID},
        {"seg_orange_cone", SMALL_ORANGE_CONE_ID},
        {"seg_small_orange_cone", SMALL_ORANGE_CONE_ID},
        {"seg_large_orange_cone", BIG_ORANGE_CONE_ID},
        {"seg_big_orange_cone", BIG_ORANGE_CONE_ID},
        {"seg_unknown_cone", IGNORE_ID}
    };

    std::string normalized_name = class_name;
    std::transform(normalized_name.begin(), normalized_name.end(), normalized_name.begin(), [](unsigned char character) { return static_cast<char>(std::tolower(character)); });

    const auto iterator = class_ids.find(normalized_name);

    if (iterator == class_ids.end()){
        throw std::invalid_argument("Unknown FSOCO class: " + class_name);
    }

    return iterator->second;
}

    // Replaces json null values outside strings so OpenCV FileStorage can parse the file.
    std::string replace_json_nulls(const std::string& json_text){
        std::string result;
        bool inside_string = false;
        bool escaped = false;

        for (std::size_t index = 0; index < json_text.size(); ++index){
            const char character = json_text[index];

            if (escaped){
                result += character;
                escaped = false;
                continue;
            }

            if (character == '\\' && inside_string){
                result += character;
                escaped = true;
                continue;
            }

            if (character == '"') {
                inside_string = !inside_string;
                result += character;
                continue;
            }

            if (!inside_string && json_text.compare(index, 4, "null") == 0){
                result += "0";
                index += 3;
                continue;
            }

            result += character;
        }

        return result;
    }


    // Reads and parses a json annotation from disk.
    cv::FileStorage open_annotation(const std::filesystem::path& annotation_path){
        if (!std::filesystem::is_regular_file(annotation_path)){
            throw std::runtime_error("Annotation file not found: " + annotation_path.string());
        }

        std::ifstream file(annotation_path);

        if (!file.is_open()){
            throw std::runtime_error("Could not open annotation: " + annotation_path.string());
        }

        std::stringstream buffer;
        buffer << file.rdbuf();

        const std::string json_text = replace_json_nulls(buffer.str());

        cv::FileStorage annotation(json_text, cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);

        if (!annotation.isOpened()){
            throw std::runtime_error("Could not parse annotation: " + annotation_path.string());
        }

        return annotation;
    }

// Decodes the base64 payload used by bitmap annotations.
std::vector<unsigned char> decode_base64(const std::string& input){
    static const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::array<int, 256> lookup;
    lookup.fill(-1);

    for (std::size_t index = 0; index < alphabet.size(); ++index){
        lookup[static_cast<unsigned char>(alphabet[index])] = static_cast<int>(index);
    }

    std::vector<unsigned char> output;
    int value = 0;
    int bits = -8;

    for (unsigned char character : input){
        if (character == '='){
            break;
        }

        if (lookup[character] < 0){
            throw std::runtime_error("Invalid Base64 bitmap data.");
        }

        value = (value << 6) + lookup[character];
        bits += 6;

        if (bits >= 0) {
            output.push_back(static_cast<unsigned char>((value >> bits) & 0xFF));
            bits -= 8;
        }
    }

    return output;
}

// Decompresses the zlib-compressed bitmap payload.
std::vector<unsigned char> decompress_zlib(const std::vector<unsigned char>& compressed_data){
    z_stream stream{};
    stream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(compressed_data.data()));
    stream.avail_in = static_cast<uInt>(compressed_data.size());

    if (inflateInit(&stream) != Z_OK){
        throw std::runtime_error("Could not initialize zlib.");
    }

    std::vector<unsigned char> output;
    std::array<unsigned char, 16384> buffer{};
    int status = Z_OK;

    while (status != Z_STREAM_END){
        stream.next_out = buffer.data();
        stream.avail_out = static_cast<uInt>(buffer.size());
        status = inflate(&stream, Z_NO_FLUSH);

        if (status != Z_OK && status != Z_STREAM_END){
            inflateEnd(&stream);
            throw std::runtime_error("Could not decompress bitmap data.");
        }

        const std::size_t produced = buffer.size() - stream.avail_out;
        output.insert(output.end(), buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(produced));
    }

    inflateEnd(&stream);
    return output;
}

// Decodes a bitmap annotation into an OpenCV mask.
cv::Mat decode_bitmap(const std::string& bitmap_data){
    const std::vector<unsigned char> compressed_data = decode_base64(bitmap_data);
    const std::vector<unsigned char> image_data = decompress_zlib(compressed_data);

    if (image_data.empty()){
        throw std::runtime_error("Decoded bitmap data is empty.");
    }

    const cv::Mat encoded_image(1, static_cast<int>(image_data.size()), CV_8U, const_cast<unsigned char*>(image_data.data()));
    const cv::Mat decoded_image = cv::imdecode(encoded_image, cv::IMREAD_UNCHANGED);

    if (decoded_image.empty()){
        throw std::runtime_error("Could not decode bitmap image.");
    }

    cv::Mat source_mask;

    if (decoded_image.channels() == 4){
        cv::extractChannel(decoded_image, source_mask, 3);
    } else if (decoded_image.channels() == 1){
        source_mask = decoded_image;
    } else{
        throw std::runtime_error("Unexpected bitmap format.");
    }

    cv::Mat binary_mask;
    cv::compare(source_mask, 0, binary_mask, cv::CMP_GT);
    binary_mask /= 255;
    return binary_mask;
}

// Places an object-local bitmap into a full-size image mask.
cv::Mat object_to_full_mask(const cv::FileNode& object, int image_height, int image_width){
    const cv::FileNode bitmap = object["bitmap"];
    const cv::FileNode origin = bitmap["origin"];

    if (bitmap.empty() || bitmap["data"].empty() || origin.empty() || origin.size() != 2){
        throw std::runtime_error("Invalid bitmap annotation.");
    }

    const cv::Mat local_mask = decode_bitmap(static_cast<std::string>(bitmap["data"]));
    const int origin_x = static_cast<int>(origin[0]);
    const int origin_y = static_cast<int>(origin[1]);

    const int x_start = std::max(0, origin_x);
    const int y_start = std::max(0, origin_y);
    const int x_end = std::min(image_width, origin_x + local_mask.cols);
    const int y_end = std::min(image_height, origin_y + local_mask.rows);

    cv::Mat full_mask = cv::Mat::zeros(image_height, image_width, CV_8UC1);

    if (x_start >= x_end || y_start >= y_end){
        return full_mask;
    }

    const cv::Rect source_roi(x_start - origin_x, y_start - origin_y, x_end - x_start, y_end - y_start);
    const cv::Rect destination_roi(x_start, y_start, x_end - x_start, y_end - y_start);
    local_mask(source_roi).copyTo(full_mask(destination_roi));

    return full_mask;
}

// Extracts the tight bounding box of the non-zero mask pixels.
std::optional<std::array<int, 4>> mask_to_bbox(const cv::Mat& mask){
    std::vector<cv::Point> points;
    cv::findNonZero(mask, points);

    if (points.empty()){
        return std::nullopt;
    }

    const cv::Rect rectangle = cv::boundingRect(points);
    return std::array<int, 4>{rectangle.x, rectangle.y, rectangle.x + rectangle.width, rectangle.y + rectangle.height};
}

// Converts class IDs into BGR colors for visualization.
cv::Mat colorize_mask(const cv::Mat& mask){
    validate_mask(mask);
    cv::Mat color_mask(mask.rows, mask.cols, CV_8UC3);

    for (int row = 0; row < mask.rows; ++row){
        for (int column = 0; column < mask.cols; ++column){
            const int class_id = static_cast<int>(mask.at<unsigned char>(row, column));
            const cv::Scalar color = class_color_bgr(class_id);
            color_mask.at<cv::Vec3b>(row, column) = cv::Vec3b(static_cast<unsigned char>(color[0]), static_cast<unsigned char>(color[1]), static_cast<unsigned char>(color[2]));
        }
    }

    return color_mask;
}

}

// Checks whether an identifier belongs to one of the cone classes.
bool is_cone_class_id(int class_id){
    return class_id >= YELLOW_CONE_ID && class_id <= BIG_ORANGE_CONE_ID;
}

// Converts an internal class ID into a readable class name.
std::string class_id_to_name(int class_id){
    switch (class_id) {
        case BACKGROUND_ID:
            return "background";
        case YELLOW_CONE_ID:
            return "yellow_cone";
        case BLUE_CONE_ID:
            return "blue_cone";
        case SMALL_ORANGE_CONE_ID:
            return "small_orange_cone";
        case BIG_ORANGE_CONE_ID:
            return "big_orange_cone";
        case IGNORE_ID:
            return "ignore";
        default:
            throw std::invalid_argument("Invalid class ID: " + std::to_string(class_id));
    }
}

// Returns the BGR color associated with a class.
cv::Scalar class_color_bgr(int class_id){
    switch (class_id){
        case BACKGROUND_ID:
            return cv::Scalar(0, 0, 0);
        case YELLOW_CONE_ID:
            return cv::Scalar(0, 255, 255);
        case BLUE_CONE_ID:
            return cv::Scalar(255, 0, 0);
        case SMALL_ORANGE_CONE_ID:
            return cv::Scalar(0, 165, 255);
        case BIG_ORANGE_CONE_ID:
            return cv::Scalar(255, 0, 255);
        case IGNORE_ID:
            return cv::Scalar(255, 255, 255);
        default:
            throw std::invalid_argument("Invalid class ID: " + std::to_string(class_id));
    }
}

// Loads a color image and validates that decoding succeeded.
cv::Mat load_image(const std::filesystem::path& image_path){
    if (!std::filesystem::is_regular_file(image_path)){
        throw std::runtime_error("Image file not found: " + image_path.string());
    }

    const cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);

    if (image.empty()){
        throw std::runtime_error("Could not decode image: " + image_path.string());
    }

    return image;
}

// Matches test images and annotations by filename stem.
std::vector<DatasetPair> find_test_pairs(const std::filesystem::path& image_dir, const std::filesystem::path& annotation_dir){
    if (!std::filesystem::is_directory(image_dir) || !std::filesystem::is_directory(annotation_dir)){
        throw std::runtime_error("Test image or annotation directory not found.");
    }

    std::map<std::string, std::filesystem::path> images;

    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(image_dir)){
        if (entry.is_regular_file() && is_supported_image(entry.path())){
            images[entry.path().filename().string()] = entry.path();
        }
    }

    std::vector<DatasetPair> pairs;

    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(annotation_dir)){
        if (!entry.is_regular_file() || entry.path().extension() != ".json"){
            continue;
        }

        std::string image_name = entry.path().filename().string();
        image_name.erase(image_name.size() - 5);

        const auto image_iterator = images.find(image_name);

        if (image_iterator == images.end()){
            throw std::runtime_error("Missing image for annotation: " + entry.path().filename().string());
        }

        pairs.push_back({image_iterator->second, entry.path()});
    }

    std::sort(pairs.begin(), pairs.end(), [](const DatasetPair& first, const DatasetPair& second) { return first.image_path.string() < second.image_path.string(); });

    if (pairs.empty()){
        throw std::runtime_error("No test image/annotation pairs found.");
    }

    return pairs;
}

// Builds the semantic mask and bounding boxes from one fsoco annotation.
GroundTruth load_ground_truth(const std::filesystem::path& annotation_path, int image_height, int image_width){
    validate_image_size(image_height, image_width);

    cv::FileStorage annotation = open_annotation(annotation_path);
    const cv::FileNode objects = annotation["objects"];

    GroundTruth ground_truth;
    ground_truth.semantic_mask = cv::Mat::zeros(image_height, image_width, CV_8UC1);

    if (objects.empty()){
        return ground_truth;
    }

    for (const cv::FileNode& object : objects){
        if (static_cast<std::string>(object["geometryType"]) != "bitmap" || object["classTitle"].empty()) {
            continue;
        }

        const int class_id = class_name_to_id(static_cast<std::string>(object["classTitle"]));
        const cv::Mat object_mask = object_to_full_mask(object, image_height, image_width);
        ground_truth.semantic_mask.setTo(class_id, object_mask);

        if (class_id == IGNORE_ID){
            continue;
        }

        const std::optional<std::array<int, 4>> coordinates = mask_to_bbox(object_mask);

        if (!coordinates.has_value()){
            continue;
        }

        Box box;
        box.x_min = coordinates.value()[0];
        box.y_min = coordinates.value()[1];
        box.x_max = coordinates.value()[2];
        box.y_max = coordinates.value()[3];
        box.class_id = class_id;
        box.score = std::nullopt;
        ground_truth.boxes.push_back(box);
    }

    return ground_truth;
}

// Blends the colored semantic mask with the original image.
cv::Mat create_overlay(const cv::Mat& image_bgr, const cv::Mat& mask, double alpha){
    if (image_bgr.empty() || image_bgr.channels() != 3){
        throw std::invalid_argument("Image must be a non-empty three-channel image.");
    }

    validate_mask(mask);

    if (image_bgr.size() != mask.size()){
        throw std::invalid_argument("Image and mask sizes must match.");
    }

    if (alpha < 0.0 || alpha > 1.0){
        throw std::invalid_argument("Alpha must be in the range [0, 1].");
    }

    const cv::Mat color_mask = colorize_mask(mask);

    cv::Mat blended;
    cv::addWeighted(image_bgr, 1.0 - alpha, color_mask, alpha, 0.0, blended);

    cv::Mat foreground;
    cv::compare(mask, BACKGROUND_ID, foreground, cv::CMP_NE);

    cv::Mat overlay = image_bgr.clone();
    blended.copyTo(overlay, foreground);
    return overlay;
}
