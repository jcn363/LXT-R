use ltx_guidance::{PerturbationConfig, PerturbationKind, StgPerturbationConfig};

#[test]
fn test_perturbation_config_debug() {
    let cfg = PerturbationConfig::default();
    let debug_str = format!("{:?}", cfg);
    assert!(debug_str.contains("PerturbationConfig"));
    assert!(debug_str.contains("Gaussian"));
}

#[test]
fn test_perturbation_config_clone() {
    let cfg = PerturbationConfig {
        scale: 0.5,
        kind: PerturbationKind::Uniform,
        n_steps: 3,
    };
    let cfg2 = cfg.clone();
    assert_eq!(cfg.scale, cfg2.scale);
    assert_eq!(cfg.kind, cfg2.kind);
    assert_eq!(cfg.n_steps, cfg2.n_steps);
}

#[test]
fn test_perturbation_kind_equality() {
    assert_eq!(PerturbationKind::Gaussian, PerturbationKind::Gaussian);
    assert_ne!(PerturbationKind::Gaussian, PerturbationKind::Uniform);
    assert_ne!(PerturbationKind::Uniform, PerturbationKind::Dropout);
}

#[test]
fn test_perturbation_config_non_default() {
    let cfg = PerturbationConfig {
        scale: 0.3,
        kind: PerturbationKind::Dropout,
        n_steps: 5,
    };
    assert!((cfg.scale - 0.3).abs() < 1e-9);
    assert_eq!(cfg.kind, PerturbationKind::Dropout);
    assert_eq!(cfg.n_steps, 5);
}

#[test]
fn test_stg_config_non_default() {
    let cfg = StgPerturbationConfig {
        spatial: false,
        temporal: true,
        spatial_scale: 0.1,
        temporal_scale: 0.2,
    };
    assert!(!cfg.spatial);
    assert!(cfg.temporal);
    assert!((cfg.spatial_scale - 0.1).abs() < 1e-9);
    assert!((cfg.temporal_scale - 0.2).abs() < 1e-9);
}

#[test]
fn test_stg_config_clone() {
    let cfg = StgPerturbationConfig::default();
    let cfg2 = cfg.clone();
    assert_eq!(cfg.spatial, cfg2.spatial);
    assert_eq!(cfg.temporal, cfg2.temporal);
}

#[test]
fn test_perturbation_config_deserialize() {
    let json = r#"{"scale": 0.5, "kind": "Uniform", "n_steps": 2}"#;
    let cfg: PerturbationConfig = serde_json::from_str(json).unwrap();
    assert!((cfg.scale - 0.5).abs() < 1e-9);
    assert_eq!(cfg.kind, PerturbationKind::Uniform);
    assert_eq!(cfg.n_steps, 2);
}

#[test]
fn test_stg_config_deserialize() {
    let json =
        r#"{"spatial": true, "temporal": false, "spatial_scale": 0.1, "temporal_scale": 0.0}"#;
    let cfg: StgPerturbationConfig = serde_json::from_str(json).unwrap();
    assert!(cfg.spatial);
    assert!(!cfg.temporal);
}
