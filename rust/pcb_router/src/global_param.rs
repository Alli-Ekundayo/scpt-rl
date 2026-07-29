
pub const SQRT2: f64 = std::f64::consts::SQRT_2;
pub const TAN22_5: f64 = 0.41421356237309503; // tan(22.5 degrees)

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum VerboseLevel {
    NotSet = 0,
    Debug = 10,
    Info = 20,
    Warning = 30,
    Error = 40,
    Critical = 50,
}

/// Global routing parameters (mirrors C++ GlobalParam static class).
#[derive(Debug, Clone)]
pub struct GlobalParam {
    pub layer_num: i32,
    pub epsilon: f64,
    pub mode_90_degree: bool,

    // BoardGrid costs
    pub diagonal_cost: f64,
    pub wirelength_cost: f64,
    pub layer_change_cost: f64,

    // Obstacle costs
    pub via_insertion_cost: f64,
    pub trace_basic_cost: f64,
    pub pin_obstacle_cost: f64,

    // Step sizes of obstacle cost
    pub step_via_obs_cost: f64,
    pub step_tra_obs_cost: f64,

    // Other costs
    pub via_touch_boundary_cost: f64,
    pub trace_touch_boundary_cost: f64,
    pub via_forbidden_cost: f64,
    pub obstacle_curve_param: f64,

    // Grid setup
    pub input_scale: u32,
    pub enlarge_boundary: u32,
    pub grid_factor: f32,

    // Routing options
    pub via_under_pad: bool,
    pub use_micro_via: bool,
    pub allow_via_for_routing: bool,
    pub curving_obstacle_cost: bool,
    pub num_rip_up_reroute_iteration: u32,

    // Output
    pub output_precision: i32,
    pub output_folder: String,
    pub output_debugging_kicad_file: bool,
    pub output_debugging_grid_values_py_file: bool,
    pub output_stacked_micro_vias: bool,

    // Log
    pub log_folder: String,
    pub verbose_level: VerboseLevel,

    pub seed: i32,
}

impl Default for GlobalParam {
    fn default() -> Self {
        GlobalParam {
            layer_num: 3,
            epsilon: 1e-14,
            mode_90_degree: true,

            diagonal_cost: SQRT2,
            wirelength_cost: 1.0,
            layer_change_cost: 10.0,

            // Obstacle costs — higher values discourage routing near existing features
            via_insertion_cost: 100.0,       // penalty for inserting a via
            trace_basic_cost: 50.0,          // base cost per unit of trace near obstacles
            pin_obstacle_cost: 1000.0,       // penalty for routing near pins

            step_via_obs_cost: 0.0,
            step_tra_obs_cost: 0.0,

            // Boundary and forbidden costs
            via_touch_boundary_cost: 1000.0,
            trace_touch_boundary_cost: 100_000.0, // effectively forbids traces at boundary
            via_forbidden_cost: 2000.0,
            obstacle_curve_param: 10_000.0,       // curvature parameter for obstacle cost function

            // Grid setup
            input_scale: 10,                // grid cells per mm (10 = 0.1mm resolution)
            enlarge_boundary: 0,
            grid_factor: 0.1,               // mm per grid cell (= 1/input_scale)

            via_under_pad: false,
            use_micro_via: true,
            allow_via_for_routing: true,
            curving_obstacle_cost: true,
            num_rip_up_reroute_iteration: 5,

            output_precision: 5,
            output_folder: "output".to_string(),
            output_debugging_kicad_file: true,
            output_debugging_grid_values_py_file: true,
            output_stacked_micro_vias: true,

            log_folder: "log".to_string(),
            verbose_level: VerboseLevel::Debug,

            // Arbitrary fixed seed for deterministic routing
            seed: 1_470_295_829,
        }
    }
}

impl GlobalParam {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_grid_scale(&mut self, scale: u32) {
        self.input_scale = scale;
        self.grid_factor = 1.0 / scale as f32;
    }

    pub fn db_length_to_grid_length_ceil(&self, db_length: f64) -> i32 {
        (db_length * self.input_scale as f64).ceil() as i32
    }

    pub fn db_length_to_grid_length_floor(&self, db_length: f64) -> i32 {
        (db_length * self.input_scale as f64).floor() as i32
    }

    pub fn db_length_to_grid_length(&self, db_length: f64) -> f64 {
        db_length * self.input_scale as f64
    }

    pub fn grid_length_to_db_length(&self, grid_length: f64) -> f64 {
        grid_length / self.input_scale as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params() {
        let p = GlobalParam::default();
        assert_eq!(p.layer_num, 3);
        assert!((p.diagonal_cost - SQRT2).abs() < 1e-10);
        assert_eq!(p.input_scale, 10);
        assert!((p.grid_factor - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_set_grid_scale() {
        let mut p = GlobalParam::default();
        p.set_grid_scale(20);
        assert_eq!(p.input_scale, 20);
        assert!((p.grid_factor - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_db_to_grid_conversion() {
        let p = GlobalParam::default();
        assert_eq!(p.db_length_to_grid_length_ceil(1.05), 11);
        assert_eq!(p.db_length_to_grid_length_floor(1.05), 10);
        assert!((p.db_length_to_grid_length(2.0) - 20.0).abs() < 1e-10);
        assert!((p.grid_length_to_db_length(20.0) - 2.0).abs() < 1e-10);
    }
}
