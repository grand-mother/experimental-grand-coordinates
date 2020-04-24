#include <stdlib.h>

#include "coordinates.h"


static struct grand_frame _ecef = {
        {0, 0, 0},
        {1, 0, 0, 0, 1, 0, 0, 0, 1}
};

struct grand_frame * GRAND_ECEF = &_ecef;


void grand_coordinates_array_create(
    struct grand_coordinates_array ** coordinates,
    enum grand_coordinates_type type,
    enum grand_coordinates_system system,
    struct grand_frame * frame,
    long size)
{
        if (size < 0) size = 0;

        struct grand_coordinates_array * c = malloc(
            sizeof(*c) + 3 * size * sizeof(c->data[0]));
        *coordinates = c;

        if (c != NULL) {
                c->type = type;
                c->system = system;
                c->frame = frame;
                c->size = size;
                c->data = (size == 0) ? NULL : c->buffer;
        }
}


void grand_coordinates_array_destroy(
    struct grand_coordinates_array ** coordinates)
{
        if (coordinates == NULL) return;

        free(*coordinates);
        *coordinates = NULL;
}


void grand_coordinates_array_resize(
    struct grand_coordinates_array ** coordinates, long size)
{
        if (coordinates == NULL) return;
        if (size < 0) size = 0;

        struct grand_coordinates_array * c = realloc(*coordinates,
            sizeof(*c) + 3 * size * sizeof(c->data[0]));
        if (c != NULL) {
                *coordinates = c;
                c->data = (size == 0) ? NULL : c->buffer;
                c->size = size;
        }
}
