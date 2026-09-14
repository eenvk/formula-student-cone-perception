//Renzi

#ifndef BOX_H
#define BOX_H

#include <optional>

// Bounding box representation used by detection and evaluation modules.
struct Box {
    int x_min = 0;
    int y_min = 0;
    int x_max = 0;
    int y_max = 0;
    int class_id = 0;
    std::optional<double> score = std::nullopt;
};

#endif
