/// 2-D integer point, equivalent to C++ `Point_2D<int>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Point2D {
    pub x: i32,
    pub y: i32,
}

impl Point2D {
    pub fn new(x: i32, y: i32) -> Self {
        Point2D { x, y }
    }
}

/// 2-D floating-point point, equivalent to C++ `Point_2D<double>`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Point2Df {
    pub x: f64,
    pub y: f64,
}

impl Point2Df {
    pub fn new(x: f64, y: f64) -> Self {
        Point2Df { x, y }
    }

    pub fn distance(a: &Point2Df, b: &Point2Df) -> f64 {
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Simple polygon as ordered list of `Point2Df` vertices.
#[derive(Debug, Clone, Default)]
pub struct Polygon {
    pub vertices: Vec<Point2Df>,
}

impl Polygon {
    pub fn new() -> Self {
        Polygon { vertices: Vec::new() }
    }

    pub fn from_vertices(vertices: Vec<Point2Df>) -> Self {
        Polygon { vertices }
    }

    /// Bounding box (min, max).
    pub fn bounding_box(&self) -> Option<(Point2Df, Point2Df)> {
        if self.vertices.is_empty() {
            return None;
        }
        let mut min = self.vertices[0];
        let mut max = self.vertices[0];
        for v in &self.vertices {
            if v.x < min.x { min.x = v.x; }
            if v.y < min.y { min.y = v.y; }
            if v.x > max.x { max.x = v.x; }
            if v.y > max.y { max.y = v.y; }
        }
        Some((min, max))
    }

    /// Point-in-polygon test (ray casting).
    pub fn contains(&self, p: &Point2Df) -> bool {
        let n = self.vertices.len();
        if n < 3 {
            return false;
        }
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let vi = &self.vertices[i];
            let vj = &self.vertices[j];
            if ((vi.y > p.y) != (vj.y > p.y))
                && (p.x < (vj.x - vi.x) * (p.y - vi.y) / (vj.y - vi.y) + vi.x)
            {
                inside = !inside;
            }
            j = i;
        }
        inside
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point2d() {
        let p = Point2D::new(3, 4);
        assert_eq!(p.x, 3);
        assert_eq!(p.y, 4);
    }

    #[test]
    fn test_point2df_distance() {
        let a = Point2Df::new(0.0, 0.0);
        let b = Point2Df::new(3.0, 4.0);
        assert!((Point2Df::distance(&a, &b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_polygon_contains() {
        // Unit square
        let poly = Polygon::from_vertices(vec![
            Point2Df::new(0.0, 0.0),
            Point2Df::new(1.0, 0.0),
            Point2Df::new(1.0, 1.0),
            Point2Df::new(0.0, 1.0),
        ]);
        assert!(poly.contains(&Point2Df::new(0.5, 0.5)));
        assert!(!poly.contains(&Point2Df::new(2.0, 0.5)));
    }

    #[test]
    fn test_polygon_bounding_box() {
        let poly = Polygon::from_vertices(vec![
            Point2Df::new(1.0, 2.0),
            Point2Df::new(4.0, 0.0),
            Point2Df::new(3.0, 5.0),
        ]);
        let (min, max) = poly.bounding_box().unwrap();
        assert!((min.x - 1.0).abs() < 1e-10);
        assert!((min.y - 0.0).abs() < 1e-10);
        assert!((max.x - 4.0).abs() < 1e-10);
        assert!((max.y - 5.0).abs() < 1e-10);
    }
}
