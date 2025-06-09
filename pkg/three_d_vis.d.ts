/* tslint:disable */
/* eslint-disable */
export class CubeRenderer {
  free(): void;
  constructor(canvas: HTMLCanvasElement);
  static new_force_webgl(canvas: HTMLCanvasElement): Promise<CubeRenderer>;
  static new_with_background(canvas: HTMLCanvasElement, background_color: string): Promise<CubeRenderer>;
  static new_force_webgl_with_background(canvas: HTMLCanvasElement, background_color: string): Promise<CubeRenderer>;
  resize(width: number, height: number): void;
  zoom(delta: number): void;
  pan(delta_x: number, delta_y: number): void;
  rotate(delta_x: number, delta_y: number): void;
  render(): void;
  get_performance_snapshot(): PerformanceSnapshot | undefined;
  get_visible_objects(): number;
  get_total_objects(): number;
  get_culling_ratio(): number;
  create_test_objects(count: number): void;
  add_object(x: number, y: number, z: number, radius: number): void;
  enable_instancing_demo_with_size(grid_size: number): void;
  enable_gizmo(): void;
  disable_gizmo(): void;
  is_gizmo_enabled(): boolean;
}
export class PerformanceSnapshot {
  private constructor();
  free(): void;
  timestamp: number;
  render_calls_per_sec: number;
  actual_renders_per_sec: number;
  dirty_ratio: number;
  frame_count: number;
  dirty_frame_count: number;
  avg_render_time_ms: number;
  object_count: number;
  edge_count: number;
  vertex_count: number;
  index_count: number;
  memory_usage_mb: number;
  scene_size_memory_mb: number;
  active_view_memory_mb: number;
  visible_objects: number;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_cuberenderer_free: (a: number, b: number) => void;
  readonly cuberenderer_new: (a: any) => any;
  readonly cuberenderer_new_force_webgl: (a: any) => any;
  readonly cuberenderer_new_with_background: (a: any, b: number, c: number) => any;
  readonly cuberenderer_new_force_webgl_with_background: (a: any, b: number, c: number) => any;
  readonly cuberenderer_resize: (a: number, b: number, c: number) => void;
  readonly cuberenderer_zoom: (a: number, b: number) => void;
  readonly cuberenderer_pan: (a: number, b: number, c: number) => void;
  readonly cuberenderer_rotate: (a: number, b: number, c: number) => void;
  readonly cuberenderer_render: (a: number) => [number, number];
  readonly cuberenderer_get_performance_snapshot: (a: number) => number;
  readonly cuberenderer_get_visible_objects: (a: number) => number;
  readonly cuberenderer_get_total_objects: (a: number) => number;
  readonly cuberenderer_get_culling_ratio: (a: number) => number;
  readonly cuberenderer_create_test_objects: (a: number, b: number) => void;
  readonly cuberenderer_add_object: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly cuberenderer_enable_instancing_demo_with_size: (a: number, b: number) => void;
  readonly cuberenderer_enable_gizmo: (a: number) => void;
  readonly cuberenderer_disable_gizmo: (a: number) => void;
  readonly cuberenderer_is_gizmo_enabled: (a: number) => number;
  readonly __wbg_performancesnapshot_free: (a: number, b: number) => void;
  readonly __wbg_get_performancesnapshot_timestamp: (a: number) => number;
  readonly __wbg_set_performancesnapshot_timestamp: (a: number, b: number) => void;
  readonly __wbg_get_performancesnapshot_render_calls_per_sec: (a: number) => number;
  readonly __wbg_set_performancesnapshot_render_calls_per_sec: (a: number, b: number) => void;
  readonly __wbg_get_performancesnapshot_actual_renders_per_sec: (a: number) => number;
  readonly __wbg_set_performancesnapshot_actual_renders_per_sec: (a: number, b: number) => void;
  readonly __wbg_get_performancesnapshot_dirty_ratio: (a: number) => number;
  readonly __wbg_set_performancesnapshot_dirty_ratio: (a: number, b: number) => void;
  readonly __wbg_get_performancesnapshot_frame_count: (a: number) => number;
  readonly __wbg_set_performancesnapshot_frame_count: (a: number, b: number) => void;
  readonly __wbg_get_performancesnapshot_dirty_frame_count: (a: number) => number;
  readonly __wbg_set_performancesnapshot_dirty_frame_count: (a: number, b: number) => void;
  readonly __wbg_get_performancesnapshot_avg_render_time_ms: (a: number) => number;
  readonly __wbg_set_performancesnapshot_avg_render_time_ms: (a: number, b: number) => void;
  readonly __wbg_get_performancesnapshot_object_count: (a: number) => number;
  readonly __wbg_set_performancesnapshot_object_count: (a: number, b: number) => void;
  readonly __wbg_get_performancesnapshot_edge_count: (a: number) => number;
  readonly __wbg_set_performancesnapshot_edge_count: (a: number, b: number) => void;
  readonly __wbg_get_performancesnapshot_vertex_count: (a: number) => number;
  readonly __wbg_set_performancesnapshot_vertex_count: (a: number, b: number) => void;
  readonly __wbg_get_performancesnapshot_index_count: (a: number) => number;
  readonly __wbg_set_performancesnapshot_index_count: (a: number, b: number) => void;
  readonly __wbg_get_performancesnapshot_memory_usage_mb: (a: number) => number;
  readonly __wbg_set_performancesnapshot_memory_usage_mb: (a: number, b: number) => void;
  readonly __wbg_get_performancesnapshot_scene_size_memory_mb: (a: number) => number;
  readonly __wbg_set_performancesnapshot_scene_size_memory_mb: (a: number, b: number) => void;
  readonly __wbg_get_performancesnapshot_active_view_memory_mb: (a: number) => number;
  readonly __wbg_set_performancesnapshot_active_view_memory_mb: (a: number, b: number) => void;
  readonly __wbg_get_performancesnapshot_visible_objects: (a: number) => number;
  readonly __wbg_set_performancesnapshot_visible_objects: (a: number, b: number) => void;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_export_2: WebAssembly.Table;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_export_6: WebAssembly.Table;
  readonly __externref_table_dealloc: (a: number) => void;
  readonly closure819_externref_shim: (a: number, b: number, c: any) => void;
  readonly closure2257_externref_shim: (a: number, b: number, c: any, d: any) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
