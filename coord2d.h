#ifndef _COORD2D_H_
#define _COORD2D_H_

template <class T>
class coord2d {
    public:
        T x;
        T y;

        // Constructors
        coord2d() {
            this->x = T(0);
            this->y = T(0);
        }

        coord2d(const T x, const T y) {
            this->x = x;
            this->y = y;
        }

        coord2d(const T a) {
            this->x = a;
            this->y = a;
        }

        // Overload some operators
        coord2d<T>& operator=(const coord2d<T>& c) {
            this->x = c.x;
            this->y = c.y;
            return *this;
        }

        coord2d<T>& operator=(const T& a) {
            this->x = a;
            this->y = a;
            return *this;
        }

        // Addition
        coord2d<T> operator+(const coord2d<T>& c) {
            return coord2d<T>(this->x + c.x, this->y + c.y);
        }

        coord2d<T> operator+(const T& a) {
            return coord2d<T>(this->x + a, this->y + a);
        }

        coord2d<T>& operator+=(const coord2d<T> c) {
            this->x += c.x;
            this->y += c.y;
            return *this;
        }

        coord2d<T>& operator+=(const T& a) {
            this->x += a;
            this->y += a;
            return *this;
        }

        // Subtraction
        coord2d<T> operator-(const coord2d<T>& c) {
            return coord2d<T>(this->x - c.x, this->y - c.y);
        }

        coord2d<T> operator-(const T& a) {
            return coord2d<T>(this->x - a, this->y - a);
        }

        coord2d<T>& operator-=(const coord2d<T> c) {
            this->x -= c.x;
            this->y -= c.y;
            return *this;
        }

        coord2d<T>& operator -=(const T& a) {
            this->x -= a;
            this->y -= a;
            return *this;
        }

        // Multiplication
        coord2d<T> operator*(const T& a) {
            return coord2d<T>(this->x * a, this->y * a);
        }

        coord2d<T>& operator*=(const T& a) {
            this->x *= a;
            this->y *= a;
            return *this;
        }

        // Division
        coord2d<T> operator/(const T& a) {
            if (a != 0) {
                return coord2d<T>(this->x/a, this->y/a);
            }
            else {
                return *this;
            }
        }

        coord2d<T>& operator/=(const T& a) {
            if (a != 0) {
                this->x /= a;
                this->y /= a;
            }
            return *this;
        }

        // Boolean operators
        bool operator==(const coord2d& c) {
            return (this->x == c.x) && (this->y == c.y);
        }

        bool operator!=(const coord2d& c) {
            return (this->x != c.x) || (this->y != c.y);
        }

};

typedef coord2d<unsigned int> dim;
typedef coord2d<float> vector2d;

#endif