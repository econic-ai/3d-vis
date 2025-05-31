use wasm_bindgen::prelude::*;
use web_sys::Performance;
use std::collections::VecDeque;

// WASM-compatible performance timing
fn performance() -> Performance {
    web_sys::window()
        .expect("should have a window in this context")
        .performance()
        .expect("performance should be available")
}

pub fn now() -> f64 {
    performance().now()
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct PerformanceSnapshot {
    pub timestamp: f64,        // Milliseconds since session start
    pub fps: f64,              // Current FPS
    pub frame_time_ms: f64,    // Average frame time in window
    pub min_frame_time: f64,   // Min frame time in window  
    pub max_frame_time: f64,   // Max frame time in window
    pub frame_count: u32,      // Total frames since start
    
    // Camera tracking metrics
    pub camera_updates_per_sec: f64,    // How often camera changes
    pub camera_dirty_ratio: f64,        // % of frames where camera changed
    
    // Memory/Buffer tracking
    pub buffer_uploads_per_sec: f64,    // Buffer upload frequency
    pub bytes_uploaded_per_sec: f64,    // Memory bandwidth usage
    pub uniform_updates_per_sec: f64,   // Uniform buffer update frequency
    
    // Performance breakdown (in microseconds)
    pub matrix_calc_time_us: f64,       // Time spent on matrix calculations
    pub buffer_upload_time_us: f64,     // Time spent uploading buffers
    pub gpu_submit_time_us: f64,        // Time spent submitting GPU commands
    
    // Efficiency ratios
    pub cpu_utilization_ratio: f64,     // CPU work / total frame time
    pub memory_efficiency: f64,         // Useful uploads / total uploads
    
    // Geometry metrics (cached, updated only on scene changes)
    pub object_count: u32,              // Total objects in scene
    pub edge_count: u32,                // Total edges in scene
    pub vertex_count: u32,              // Total vertices in scene template
    pub index_count: u32,               // Total indices for all objects
    
    // Memory and GPU throughput
    pub memory_usage_mb: f64,           // Current GPU memory usage in MB
    pub gpu_vertices_per_sec: f64,      // Actual GPU vertex processing (visible only)
    
    // New memory breakdown metrics
    pub scene_size_memory_mb: f64,      // Total memory for all objects in scene
    pub active_view_memory_mb: f64,     // Memory for currently visible objects (post-culling)
    pub active_memory_throughput_mb_per_sec: f64, // Memory transfer rate for visible objects
}

pub struct PerformanceTracker {
    frame_times: VecDeque<(f64, f64)>, // (timestamp, duration) pairs in milliseconds
    session_start: f64,
    last_frame_start: Option<f64>,
    last_export: f64,
    total_frames: u32,
    
    // Export interval (100ms = 10Hz)
    export_interval: f64,
    
    // Camera state tracking
    last_camera_state: (f32, f32, f32, f32, f32), // (distance, pan_x, pan_y, rot_x, rot_y)
    camera_updates: VecDeque<f64>, // timestamps of camera changes
    
    // Memory/Buffer tracking  
    buffer_uploads: VecDeque<(f64, u32)>, // (timestamp, bytes) pairs
    uniform_updates: VecDeque<f64>, // timestamps of uniform buffer updates
    
    // Performance timing breakdown
    pub current_matrix_calc_time: f64,
    pub current_buffer_upload_time: f64, 
    pub current_gpu_submit_time: f64,
    
    // Running totals for efficiency calculations
    total_cpu_time: f64,
    total_gpu_time: f64,
}

impl PerformanceTracker {
    pub fn new() -> Self {
        let now = now();
        Self {
            frame_times: VecDeque::new(),
            session_start: now,
            last_frame_start: None,
            last_export: now,
            total_frames: 0,
            export_interval: 100.0, // 100ms
            last_camera_state: (5.0, 0.0, 0.0, 0.0, 0.0),
            camera_updates: VecDeque::new(),
            buffer_uploads: VecDeque::new(),
            uniform_updates: VecDeque::new(),
            current_matrix_calc_time: 0.0,
            current_buffer_upload_time: 0.0,
            current_gpu_submit_time: 0.0,
            total_cpu_time: 0.0,
            total_gpu_time: 0.0,
        }
    }
    
    pub fn start_frame(&mut self) {
        self.last_frame_start = Some(now());
    }
    
    pub fn end_frame(&mut self) -> Option<PerformanceSnapshot> {
        if let Some(frame_start) = self.last_frame_start.take() {
            let frame_end = now();
            let frame_duration = frame_end - frame_start;
            
            // Add this frame to our rolling window
            self.frame_times.push_back((frame_start, frame_duration));
            self.total_frames += 1;
            
            // Remove frames older than 1 second for rolling calculation
            let one_second_ago = frame_end - 1000.0; // 1 second = 1000ms
            while let Some(&(timestamp, _)) = self.frame_times.front() {
                if timestamp < one_second_ago {
                    self.frame_times.pop_front();
                } else {
                    break;
                }
            }
            
            // Check if we should export a snapshot
            if frame_end - self.last_export >= self.export_interval {
                self.last_export = frame_end;
                return Some(self.create_snapshot(frame_end));
            }
        }
        None
    }
    
    pub fn create_snapshot(&self, now: f64) -> PerformanceSnapshot {
        if self.frame_times.is_empty() {
            return PerformanceSnapshot {
                timestamp: now - self.session_start,
                fps: 0.0,
                frame_time_ms: 0.0,
                min_frame_time: 0.0,
                max_frame_time: 0.0,
                frame_count: self.total_frames,
                camera_updates_per_sec: 0.0,
                camera_dirty_ratio: 0.0,
                buffer_uploads_per_sec: 0.0,
                bytes_uploaded_per_sec: 0.0,
                uniform_updates_per_sec: 0.0,
                matrix_calc_time_us: 0.0,
                buffer_upload_time_us: 0.0,
                gpu_submit_time_us: 0.0,
                cpu_utilization_ratio: 0.0,
                memory_efficiency: 0.0,
                object_count: 0,
                edge_count: 0,
                vertex_count: 0,
                index_count: 0,
                memory_usage_mb: 0.0,
                gpu_vertices_per_sec: 0.0,
                scene_size_memory_mb: 0.0,
                active_view_memory_mb: 0.0,
                active_memory_throughput_mb_per_sec: 0.0,
            };
        }
        
        // Calculate metrics from rolling window (last 1 second)
        let window_size = self.frame_times.len() as f64;
        let fps = window_size; // frames in 1 second = FPS
        
        let durations: Vec<f64> = self.frame_times
            .iter()
            .map(|(_, duration)| *duration)
            .collect();
        
        let avg_frame_time = durations.iter().sum::<f64>() / window_size;
        let min_frame_time = durations.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_frame_time = durations.iter().fold(0.0f64, |a, &b| a.max(b));
        
        // Calculate camera update metrics
        let one_second_ago = now - 1000.0;
        let recent_camera_updates = self.camera_updates.iter()
            .filter(|&&timestamp| timestamp > one_second_ago)
            .count() as f64;
        let camera_updates_per_sec = recent_camera_updates;
        let camera_dirty_ratio = if window_size > 0.0 { recent_camera_updates / window_size } else { 0.0 };
        
        // Calculate buffer metrics
        let recent_buffer_uploads = self.buffer_uploads.iter()
            .filter(|(timestamp, _)| *timestamp > one_second_ago)
            .count() as f64;
        let total_bytes_uploaded: u32 = self.buffer_uploads.iter()
            .filter(|(timestamp, _)| *timestamp > one_second_ago)
            .map(|(_, bytes)| *bytes)
            .sum();
        let bytes_uploaded_per_sec = total_bytes_uploaded as f64;
        
        let recent_uniform_updates = self.uniform_updates.iter()
            .filter(|&&timestamp| timestamp > one_second_ago)
            .count() as f64;
        
        // Calculate efficiency ratios
        let total_frame_time = avg_frame_time;
        let cpu_time = self.current_matrix_calc_time + self.current_buffer_upload_time;
        let cpu_utilization_ratio = if total_frame_time > 0.0 { cpu_time / total_frame_time } else { 0.0 };
        
        let memory_efficiency = if recent_buffer_uploads > 0.0 { 
            recent_uniform_updates / recent_buffer_uploads 
        } else { 
            1.0 
        };
        
        PerformanceSnapshot {
            timestamp: now - self.session_start,
            fps,
            frame_time_ms: avg_frame_time,
            min_frame_time,
            max_frame_time,
            frame_count: self.total_frames,
            camera_updates_per_sec,
            camera_dirty_ratio,
            buffer_uploads_per_sec: recent_buffer_uploads,
            bytes_uploaded_per_sec,
            uniform_updates_per_sec: recent_uniform_updates,
            matrix_calc_time_us: self.current_matrix_calc_time * 1000.0, // Convert to μs
            buffer_upload_time_us: self.current_buffer_upload_time * 1000.0, // Convert to μs  
            gpu_submit_time_us: self.current_gpu_submit_time * 1000.0, // Convert to μs
            cpu_utilization_ratio,
            memory_efficiency,
            object_count: 0,
            edge_count: 0,
            vertex_count: 0,
            index_count: 0,
            memory_usage_mb: 0.0,
            gpu_vertices_per_sec: 0.0,
            scene_size_memory_mb: 0.0,
            active_view_memory_mb: 0.0,
            active_memory_throughput_mb_per_sec: 0.0,
        }
    }
    
    // Enhanced create_snapshot that accepts renderer data for complete metrics
    pub fn create_snapshot_with_renderer_data(
        &self, 
        now: f64,
        cached_object_count: u32,
        cached_edge_count: u32,
        cached_vertex_count: u32,
        cached_index_count: u32,
        total_memory_usage_bytes: u64,
        scene_size_memory_bytes: u64,
        active_view_memory_bytes: u64,
        visible_objects: u32,
        vertices_per_object: usize,
    ) -> PerformanceSnapshot {
        // Get base snapshot with standard performance metrics
        let mut snapshot = self.create_snapshot(now);
        
        // Add cached geometry metrics
        snapshot.object_count = cached_object_count;
        snapshot.edge_count = cached_edge_count;
        snapshot.vertex_count = cached_vertex_count;
        snapshot.index_count = cached_index_count;
        
        // Calculate memory metrics in MB
        snapshot.memory_usage_mb = total_memory_usage_bytes as f64 / (1024.0 * 1024.0);
        snapshot.scene_size_memory_mb = scene_size_memory_bytes as f64 / (1024.0 * 1024.0);
        snapshot.active_view_memory_mb = active_view_memory_bytes as f64 / (1024.0 * 1024.0);
        
        // Calculate GPU vertices per second (only visible vertices that GPU actually processes)
        snapshot.gpu_vertices_per_sec = (visible_objects as f64) * (vertices_per_object as f64) * snapshot.fps;
        
        // Calculate active memory throughput (MB/s for visible objects)
        if visible_objects > 0 && snapshot.fps > 0.0 {
            // Calculate bytes per frame for visible objects (instance data size per object)
            let bytes_per_visible_object = 32.0; // InstanceData is ~32 bytes (position + color + scale)
            let bytes_per_frame = (visible_objects as f64) * bytes_per_visible_object;
            let bytes_per_second = bytes_per_frame * snapshot.fps;
            snapshot.active_memory_throughput_mb_per_sec = bytes_per_second / (1024.0 * 1024.0);
        } else {
            snapshot.active_memory_throughput_mb_per_sec = 0.0;
        }
        
        snapshot
    }
    
    pub fn track_camera_change(&mut self, distance: f32, pan_x: f32, pan_y: f32, rot_x: f32, rot_y: f32) {
        let current_state = (distance, pan_x, pan_y, rot_x, rot_y);
        
        // Check if camera state actually changed  
        if current_state != self.last_camera_state {
            let now = now();
            self.camera_updates.push_back(now);
            self.last_camera_state = current_state;
            
            // Clean old camera updates (older than 1 second)
            let one_second_ago = now - 1000.0;
            while let Some(&timestamp) = self.camera_updates.front() {
                if timestamp < one_second_ago {
                    self.camera_updates.pop_front();
                } else {
                    break;
                }
            }
        }
    }
    
    pub fn track_buffer_upload(&mut self, bytes: u32) {
        let now = now();
        self.buffer_uploads.push_back((now, bytes));
        
        // Clean old buffer uploads (older than 1 second)
        let one_second_ago = now - 1000.0;
        while let Some(&(timestamp, _)) = self.buffer_uploads.front() {
            if timestamp < one_second_ago {
                self.buffer_uploads.pop_front();
            } else {
                break;
            }
        }
    }
    
    pub fn track_uniform_update(&mut self) {
        let now = now();
        self.uniform_updates.push_back(now);
        
        // Clean old uniform updates (older than 1 second)
        let one_second_ago = now - 1000.0;
        while let Some(&timestamp) = self.uniform_updates.front() {
            if timestamp < one_second_ago {
                self.uniform_updates.pop_front();
            } else {
                break;
            }
        }
    }
    
    pub fn has_frames(&self) -> bool {
        !self.frame_times.is_empty()
    }
} 