enum grand_coordinates_system {
        GRAND_COORDINATES_CARTESIAN = 0,
        GRAND_COORDINATES_GEODETIC,
        GRAND_COORDINATES_HORIZONTAL,
        GRAND_COORDINATES_SPHERICAL,
        GRAND_COORDINATES_UNDEFINED_SYSTEM = -1
};


enum grand_coordinates_type {
        GRAND_COORDINATES_POINT = 0,
        GRAND_COORDINATES_VECTOR,
        GRAND_COORDINATES_UNDEFINED_TYPE = -1
};


struct grand_frame {
        double origin[3];
        double basis[9];
};


struct grand_coordinates_cartesian { double x, y, z; };
struct grand_coordinates_geodetic { double latitude, longitude, height; };
struct grand_coordinates_horizontal { double azimuth, elevation, norm; };
struct grand_coordinates_spherical { double r, theta, phi; };


struct grand_coordinates_array {
        enum grand_coordinates_type type;
        enum grand_coordinates_system system;
        struct grand_frame * frame;
        long size;
        double * data;
        double buffer[];
};


void grand_coordinates_array_create(
    struct grand_coordinates_array ** coordinates,
    enum grand_coordinates_type type, enum grand_coordinates_system system,
    struct grand_frame * frame, long size);


void grand_coordinates_array_destroy(
    struct grand_coordinates_array ** coordinates);


void grand_coordinates_array_resize(
    struct grand_coordinates_array ** coordinates, long size);
