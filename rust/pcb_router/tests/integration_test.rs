//! Integration tests exercising the full routing pipeline.

use pcb_router::board_grid::BoardGrid;
use pcb_router::global_param::GlobalParam;
use pcb_router::grid_based_router::GridBasedRouter;
use pcb_router::grid_netclass::GridNetclass;
use pcb_router::kicad_parser::types::*;
use pcb_router::kicad_parser::KicadPcbDatabase;
use pcb_router::location::Location;

/// Build a minimal two-layer test board with the given pads.
fn build_test_db(pads: Vec<Pad>) -> KicadPcbDatabase {
    let mut db = KicadPcbDatabase::new();

    db.layers.push(Layer { id: 0, name: "F.Cu".into(), layer_type: LayerType::Copper });
    db.layers.push(Layer { id: 31, name: "B.Cu".into(), layer_type: LayerType::Copper });
    db.layer_id_to_name.insert(0, "F.Cu".into());
    db.layer_id_to_name.insert(31, "B.Cu".into());

    db.netclasses.push(Netclass {
        name: "Default".into(),
        description: String::new(),
        clearance: 0.2,
        trace_width: 0.25,
        via_dia: 0.8,
        via_drill: 0.4,
        uvia_dia: 0.3,
        uvia_drill: 0.15,
    });

    db.nets.push(Net { id: 1, name: "GND".into(), netclass_name: "Default".into(), pins: vec![] });

    for pad in pads {
        db.pads.push(pad);
    }
    db
}

fn make_pad(ref_name: &str, number: &str, pos: (f64, f64), net_id: i32, layers: Vec<&str>) -> Pad {
    Pad {
        reference: ref_name.into(),
        number: number.into(),
        pad_type: "smd".into(),
        shape: "rect".into(),
        position: pos,
        size: (1.0, 1.0),
        net_id,
        layers: layers.into_iter().map(|s| s.to_string()).collect(),
        instance_position: (0.0, 0.0),
    }
}

#[test]
fn test_full_route_two_pins_same_layer() {
    let pads = vec![
        make_pad("U1", "1", (10.0, 10.0), 1, vec!["F.Cu"]),
        make_pad("U1", "2", (15.0, 10.0), 1, vec!["F.Cu"]),
    ];
    let db = build_test_db(pads);

    let mut router = GridBasedRouter::new(GlobalParam::default());
    router.initialization(&db);
    router.route_all();

    let wl = router.routed_wirelength();
    assert!(wl > 0.0, "Should have positive wirelength");
    assert_eq!(router.routed_num_vias(), 0, "Same-layer route should have no vias");
}

#[test]
fn test_full_route_two_pins_different_layers() {
    let pads = vec![
        make_pad("U1", "1", (10.0, 10.0), 1, vec!["F.Cu"]),
        make_pad("U1", "2", (15.0, 10.0), 1, vec!["B.Cu"]),
    ];
    let db = build_test_db(pads);

    let mut router = GridBasedRouter::new(GlobalParam::default());
    router.initialization(&db);
    router.route_all();

    let wl = router.routed_wirelength();
    assert!(wl > 0.0, "Should have positive wirelength");
    // Cross-layer route needs at least one via
    assert!(router.routed_num_vias() >= 1, "Cross-layer route should use vias");
}

#[test]
fn test_full_route_multipin_net() {
    // Three pins on the same net
    let pads = vec![
        make_pad("U1", "1", (5.0, 5.0), 1, vec!["F.Cu"]),
        make_pad("U1", "2", (10.0, 5.0), 1, vec!["F.Cu"]),
        make_pad("U1", "3", (10.0, 10.0), 1, vec!["F.Cu"]),
    ];
    let db = build_test_db(pads);

    let mut router = GridBasedRouter::new(GlobalParam::default());
    router.initialization(&db);
    router.route_all();

    let wl = router.routed_wirelength();
    assert!(wl > 0.0, "Multipin net should have positive wirelength");
    // Should have at least 2 sub-routes (3 pins → 2 connections in spanning tree)
    let solution_paths: usize = router.best_solution.iter()
        .map(|s| s.grid_paths.len())
        .sum();
    assert!(solution_paths >= 2, "3-pin net should produce at least 2 sub-paths, got {}", solution_paths);
}

#[test]
fn test_full_route_with_obstacle() {
    // Two nets that might interfere
    let pads = vec![
        make_pad("U1", "1", (5.0, 5.0), 1, vec!["F.Cu"]),
        make_pad("U1", "2", (15.0, 5.0), 1, vec!["F.Cu"]),
        make_pad("U2", "1", (5.0, 6.0), 2, vec!["F.Cu"]),
        make_pad("U2", "2", (15.0, 6.0), 2, vec!["F.Cu"]),
    ];
    let mut db = build_test_db(pads);
    db.nets.push(Net { id: 2, name: "VCC".into(), netclass_name: "Default".into(), pins: vec![] });

    let mut router = GridBasedRouter::new(GlobalParam::default());
    router.initialization(&db);
    router.route_all();

    let wl = router.routed_wirelength();
    assert!(wl > 0.0, "Both nets should have been routed");
}

#[test]
fn test_router_setters() {
    let mut router = GridBasedRouter::new(GlobalParam::default());
    router.set_grid_scale(20);
    assert_eq!(router.params.input_scale, 20);

    router.set_num_iterations(10);
    assert_eq!(router.params.num_rip_up_reroute_iteration, 10);

    router.set_enlarge_boundary(5);
    assert_eq!(router.params.enlarge_boundary, 5);

    router.set_layer_change_weight(20.0);
    assert!((router.params.layer_change_cost - 20.0).abs() < 1e-10);

    router.set_track_obstacle_weight(100.0);
    assert!((router.params.trace_basic_cost - 100.0).abs() < 1e-10);
}

#[test]
fn test_board_grid_large_dimensions_no_overflow() {
    // Verify that large grid dimensions don't overflow in location_to_id
    let mut bg = BoardGrid::new();
    // 1000 x 1000 x 4 = 4M cells — fits in i32 math
    bg.initialize(1000, 1000, 4);

    let nc = GridNetclass::new(0, 2, 4, 8, 4, 6, 3);
    bg.add_grid_netclass(nc);

    let loc = Location::new(999, 999, 3);
    let id = bg.location_to_id(&loc);
    let back = bg.id_to_location(id);
    assert_eq!(loc, back, "Round-trip should work for large grids");
}

#[test]
fn test_kicad_parser_roundtrip_via_string() {
    let content = r#"
(kicad_pcb (version 20211014) (generator pcbnew)
  (layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
  )
  (net 0 "")
  (net 1 "GND")
  (net_class "Default" "Default net class."
    (clearance 0.2)
    (trace_width 0.25)
    (via_dia 0.8)
    (via_drill 0.4)
    (uvia_dia 0.3)
    (uvia_drill 0.1)
    (add_net "GND")
  )
  (footprint "TestPackage" (layer "F.Cu")
    (at 10 10)
    (pad "1" smd rect (at 0 0) (size 1 1) (layers "F.Cu") (net 1 "GND"))
    (pad "2" smd rect (at 5 0) (size 1 1) (layers "F.Cu") (net 1 "GND"))
  )
)
"#;
    let db = KicadPcbDatabase::from_reader(content.as_bytes()).unwrap();
    assert_eq!(db.layers.len(), 2);
    assert!(!db.netclasses.is_empty(), "Should parse legacy netclasses");
    assert_eq!(db.pads.len(), 2);
    assert_eq!(db.pads[0].net_id, 1);
}

#[test]
fn test_nan_priority_queue_handling() {
    // Verify that NaN priorities don't break the queue
    use pcb_router::location::LocationQueue;
    let mut q: LocationQueue<Location> = LocationQueue::new();
    q.push(Location::new(1, 0, 0), f32::NAN);
    q.push(Location::new(2, 0, 0), 1.0);
    q.push(Location::new(3, 0, 0), f32::INFINITY);
    q.push(Location::new(4, 0, 0), 5.0);

    // Non-NaN items should come out first (in priority order)
    let first = q.pop().unwrap();
    assert_eq!(first, Location::new(2, 0, 0), "Lowest finite priority should come first");
    let second = q.pop().unwrap();
    assert_eq!(second, Location::new(4, 0, 0));
    // INFINITY and NaN come after
    let _third = q.pop().unwrap();
    let _fourth = q.pop().unwrap();
    assert!(q.is_empty());
}
