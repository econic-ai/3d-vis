mod utils;
mod performance;
mod types;
mod camera;
mod math;

use wasm_bindgen::prelude::*;
use web_sys::{HtmlCanvasElement, console};
use wgpu::util::DeviceExt;
use cgmath::Point3;

// Re-export from modules
pub use performance::{PerformanceSnapshot, PerformanceTracker, now};
pub use types::{Vertex, Uniforms, VERTICES, INDICES, InstanceData};
pub use camera::Camera;
pub use math::{Frustum, BoundingSphere};

// Simple renderable object for demonstrating frustum culling
#[derive(Debug, Clone)]
struct RenderableObject {
    position: Point3<f32>,
    bounding_sphere: BoundingSphere,
    visible: bool, // Result of frustum culling
}

impl RenderableObject {
    fn new(position: Point3<f32>, size: f32) -> Self {
        Self {
            position,
            bounding_sphere: BoundingSphere::for_cube(position, size),
            visible: true,
        }
    }
}

#[wasm_bindgen]
pub struct CubeRenderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    width: u32,
    height: u32,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    
    // Instance buffer for GPU instancing
    instance_buffer: wgpu::Buffer,
    instance_data: Vec<InstanceData>,
    max_instances: u32,
    
    uniforms: Uniforms,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    
    // Depth buffer for 3D rendering
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    
    // Camera system
    camera: Camera,
    
    // Dirty tracking for optimization
    uniforms_dirty: bool,
    
    // Command buffer optimization - cache descriptors
    command_encoder_desc: wgpu::CommandEncoderDescriptor<'static>,
    texture_view_desc: wgpu::TextureViewDescriptor<'static>,
    
    // Background color
    background_color: wgpu::Color,
    
    // Performance tracking
    performance_tracker: PerformanceTracker,
    
    // Objects to render (for frustum culling demonstration)
    objects: Vec<RenderableObject>,
    
    // Frustum culling state
    current_frustum: Option<Frustum>,
    visible_objects: u32,
    total_objects: u32,
    
    // Cached geometry metrics (updated only on scene changes)
    cached_object_count: u32,
    cached_edge_count: u32,
    cached_vertex_count: u32,
    cached_index_count: u32,
}

#[wasm_bindgen]
impl CubeRenderer {
    fn parse_hex_color(hex: &str) -> Result<wgpu::Color, JsValue> {
        let hex = hex.trim_start_matches('#');
        if hex.len() != 6 {
            return Err(JsValue::from_str("Invalid hex color format. Expected #RRGGBB"));
        }
        
        let r = u8::from_str_radix(&hex[0..2], 16)
            .map_err(|_| JsValue::from_str("Invalid hex color format"))?;
        let g = u8::from_str_radix(&hex[2..4], 16)
            .map_err(|_| JsValue::from_str("Invalid hex color format"))?;
        let b = u8::from_str_radix(&hex[4..6], 16)
            .map_err(|_| JsValue::from_str("Invalid hex color format"))?;
        
        Ok(wgpu::Color {
            r: r as f64 / 255.0,
            g: g as f64 / 255.0,
            b: b as f64 / 255.0,
            a: 1.0,
        })
    }

    #[wasm_bindgen(constructor)]
    pub async fn new(canvas: HtmlCanvasElement) -> Result<CubeRenderer, JsValue> {
        Self::new_with_background(canvas, "#dddddd").await
    }

    #[wasm_bindgen]
    pub async fn new_force_webgl(canvas: HtmlCanvasElement) -> Result<CubeRenderer, JsValue> {
        Self::new_force_webgl_with_background(canvas, "#dddddd").await
    }

    #[wasm_bindgen]
    pub async fn new_with_background(canvas: HtmlCanvasElement, background_color: &str) -> Result<CubeRenderer, JsValue> {
        Self::new_with_backend_and_background(canvas, false, background_color).await
    }

    #[wasm_bindgen]
    pub async fn new_force_webgl_with_background(canvas: HtmlCanvasElement, background_color: &str) -> Result<CubeRenderer, JsValue> {
        Self::new_with_backend_and_background(canvas, true, background_color).await
    }

    async fn new_with_backend_and_background(canvas: HtmlCanvasElement, force_webgl: bool, background_color: &str) -> Result<CubeRenderer, JsValue> {
        utils::set_panic_hook();
        
        // Parse background color from hex string
        let bg_color = Self::parse_hex_color(background_color)?;
        
        console::log_1(&format!("Canvas width: {}, height: {}", canvas.width(), canvas.height()).into());

        let width = canvas.width();
        let height = canvas.height();

        if force_webgl {
            console::log_1(&"🔧 TESTING: Forcing WebGL backend".into());
            return Self::create_webgl_renderer(canvas, width, height, bg_color).await;
        }

        // Try WebGPU first, fall back to WebGL if it fails
        console::log_1(&"🚀 Attempting to use WebGPU backend...".into());
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions {
                gl: wgpu::GlBackendOptions {
                    gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
                    fence_behavior: wgpu::GlFenceBehavior::default(),
                },
                ..Default::default()
            },
        });

        // Clone canvas for potential fallback use
        let canvas_clone = canvas.clone();
        let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| JsValue::from_str(&format!("Failed to create surface: {:?}", e)))?;

        let adapter_result = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await;

        let adapter = match adapter_result {
            Ok(adapter) => {
                let backend = adapter.get_info().backend;
                console::log_1(&format!("✅ Adapter acquired successfully using: {:?}", backend).into());
                adapter
            }
            Err(e) => {
                console::log_1(&format!("❌ WebGPU adapter request failed: {:?}", e).into());
                console::log_1(&"🔄 Falling back to WebGL...".into());
                
                // Use the reusable WebGL function for fallback
                return Self::create_webgl_renderer(canvas_clone, width, height, bg_color).await;
            }
        };

        let result = Self::create_with_adapter_and_surface(adapter, surface, width, height, bg_color).await?;
        
        Ok(result)
    }

    async fn create_webgl_renderer(
        canvas: HtmlCanvasElement,
        width: u32,
        height: u32,
        bg_color: wgpu::Color,
    ) -> Result<CubeRenderer, JsValue> {
        let webgl_instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::GL,
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions {
                gl: wgpu::GlBackendOptions {
                    gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
                    fence_behavior: wgpu::GlFenceBehavior::default(),
                },
                ..Default::default()
            },
        });
        
        let webgl_surface = webgl_instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| JsValue::from_str(&format!("Failed to create WebGL surface: {:?}", e)))?;
        
        let webgl_adapter = webgl_instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&webgl_surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to request WebGL adapter: {:?}", e)))?;
        
        console::log_1(&format!("✅ WebGL adapter acquired successfully using: {:?}", webgl_adapter.get_info().backend).into());
        
        Self::create_with_adapter_and_surface(webgl_adapter, webgl_surface, width, height, bg_color).await
    }

    async fn create_with_adapter_and_surface(
        adapter: wgpu::Adapter,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
        bg_color: wgpu::Color,
    ) -> Result<CubeRenderer, JsValue> {
        // Log adapter information
        let adapter_info = adapter.get_info();
        let adapter_limits = adapter.limits();
        console::log_1(&format!("📊 Adapter backend: {:?}", adapter_info.backend).into());
        console::log_1(&format!("📊 Adapter limits: max_inter_stage_shader_components = {}", adapter_limits.max_inter_stage_shader_components).into());

        // Choose appropriate limits based on backend
        let device_limits = match adapter_info.backend {
            wgpu::Backend::BrowserWebGpu => {
                console::log_1(&"🚀 Using WebGPU limits".into());
                wgpu::Limits::default()
            }
            wgpu::Backend::Gl => {
                console::log_1(&"🔧 Using WebGL2 downlevel limits".into());
                wgpu::Limits::downlevel_webgl2_defaults()
            }
            _ => {
                console::log_1(&"⚠️ Unknown backend, using default limits".into());
                wgpu::Limits::default()
            }
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: device_limits,
                    memory_hints: wgpu::MemoryHints::default(),
                    ..Default::default()
                },
            )
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to create device: {:?}", e)))?;

        console::log_1(&"✅ Device created successfully".into());

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // Create uniform buffer
        let uniforms = Uniforms::new();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("uniform_bind_group_layout"),
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("uniform_bind_group"),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&uniform_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc(), InstanceData::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let num_indices = INDICES.len() as u32;

        console::log_1(&"🎉 CubeRenderer created successfully!".into());
        
        // Maximum number of instances for GPU instancing - support up to 8M cubes
        // Note: 8M instances * ~32 bytes per instance = ~256MB GPU buffer
        let max_instances = 8000000u32;
        
        // Create instance buffer for GPU instancing
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            size: (std::mem::size_of::<InstanceData>() * max_instances as usize) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create depth texture for 3D rendering
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create the renderer instance
        let renderer = Self {
            surface,
            device,
            queue,
            config,
            width,
            height,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            uniforms,
            uniform_buffer,
            uniform_bind_group,
            camera: Camera::new(width, height),
            background_color: bg_color,
            performance_tracker: PerformanceTracker::new(),
            command_encoder_desc: wgpu::CommandEncoderDescriptor {
                label: None,
                ..Default::default()
            },
            texture_view_desc: wgpu::TextureViewDescriptor::default(),
            uniforms_dirty: true,
            depth_texture,
            depth_view,
            objects: Vec::new(),
            current_frustum: None,
            visible_objects: 0,
            total_objects: 0,
            instance_buffer,
            instance_data: Vec::new(),
            max_instances,
            cached_object_count: 0,
            cached_edge_count: 0,
            cached_vertex_count: 0,
            cached_index_count: 0,
        };
        
        console::log_1(&"🎯 Renderer created with clean state (no default objects)".into());
        
        Ok(renderer)
    }

    #[wasm_bindgen]
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.width = width;
            self.height = height;
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            
            // Recreate depth buffer for new size
            self.depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth Texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.depth_view = self.depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
            
            // Update camera dimensions and mark matrices as dirty
            self.camera.resize(width, height);
            self.uniforms_dirty = true;
            
            console::log_1(&format!("Resized canvas to width: {}, height: {}", width, height).into());
        }
    }

    #[wasm_bindgen]
    pub fn zoom(&mut self, delta: f32) {
        self.camera.zoom(delta);
        self.uniforms_dirty = true;
    }

    #[wasm_bindgen]
    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        self.camera.pan(delta_x, delta_y);
        self.uniforms_dirty = true;
    }

    #[wasm_bindgen]
    pub fn rotate(&mut self, delta_x: f32, delta_y: f32) {
        self.camera.rotate(delta_x, delta_y);
        self.uniforms_dirty = true;
    }
    
    #[wasm_bindgen]
    pub fn render(&mut self) -> Result<(), JsValue> {
        // Start performance tracking for this frame
        self.performance_tracker.start_frame();
        
        // Track camera state changes for performance metrics
        let camera_state = self.camera.get_state();
        self.performance_tracker.track_camera_change(
            camera_state.0, camera_state.1, camera_state.2, camera_state.3, camera_state.4
        );
        
        // Only recalculate matrices and update uniforms if camera changed
        if self.uniforms_dirty {
            // Get the view-projection matrix from camera (with timing)
            let matrix_calc_time = if self.camera.is_dirty() {
                self.camera.update_matrices()
            } else {
                0.0
            };
            
            // Update uniforms with the combined matrix
            let view_proj_matrix = self.camera.get_view_proj_matrix();
            self.uniforms.update_view_proj(view_proj_matrix);
            
            // Extract frustum for culling
            self.current_frustum = Some(Frustum::from_view_proj_matrix(view_proj_matrix));
            
            // Record matrix calculation time
            self.performance_tracker.current_matrix_calc_time = matrix_calc_time;

            // Time buffer upload
            let buffer_start = now();
            
            self.queue.write_buffer(
                &self.uniform_buffer,
                0,
                bytemuck::cast_slice(&[self.uniforms]),
            );
            
            // Track buffer upload (uniform buffer is 64 bytes: 4x4 f32 matrix)
            self.performance_tracker.track_buffer_upload(64);
            self.performance_tracker.track_uniform_update();
            
            let buffer_end = now();
            self.performance_tracker.current_buffer_upload_time = buffer_end - buffer_start;
            
            // Mark uniforms as clean now that we've updated them
            self.uniforms_dirty = false;
        } else {
            // Camera hasn't changed, so we skip expensive operations
            self.performance_tracker.current_matrix_calc_time = 0.0;
            self.performance_tracker.current_buffer_upload_time = 0.0;
        }

        // Perform frustum culling on objects
        self.perform_frustum_culling();
        
        // Update instance data for GPU instancing (only visible objects)
        self.update_instance_data();

        // Time GPU submission (this always happens)
        let gpu_start = now();

        let output = self.surface
            .get_current_texture()
            .map_err(|e| JsValue::from_str(&format!("Failed to get surface texture: {:?}", e)))?;

        let view = output
            .texture
            .create_view(&self.texture_view_desc);

        // Use cached command encoder descriptor for optimization
        let mut encoder = self
            .device
            .create_command_encoder(&self.command_encoder_desc);

        // Optimized render pass setup with minimal allocations
        {
            // Create render pass with inline descriptor to avoid allocation
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None, // Skip label in release builds for performance
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.background_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            // Batch render pass operations for efficiency
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            
            // Use instanced drawing - render all visible objects in a single draw call
            let instance_count = self.instance_data.len() as u32;
            if instance_count > 0 {
                render_pass.draw_indexed(0..self.num_indices, 0, 0..instance_count);
            }
        }

        // Submit command buffer
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        let gpu_end = now();
        self.performance_tracker.current_gpu_submit_time = gpu_end - gpu_start;

        // End performance tracking and return snapshot if available
        // Note: We don't return the snapshot from render() to avoid affecting performance
        // JavaScript will poll for snapshots separately
        self.performance_tracker.end_frame();

        Ok(())
    }

    #[wasm_bindgen]
    pub fn get_performance_snapshot(&mut self) -> Option<PerformanceSnapshot> {
        // Force create a snapshot for JavaScript to consume
        // This is called periodically from JavaScript, not every frame
        if self.performance_tracker.has_frames() {
            let now = now();
            let snapshot = self.performance_tracker.create_snapshot_with_renderer_data(
                now,
                self.cached_object_count,
                self.cached_edge_count,
                self.cached_vertex_count,
                self.cached_index_count,
                self.calculate_current_memory_usage(),
                self.calculate_scene_size_memory_bytes(),
                self.calculate_active_view_memory_bytes(),
                self.visible_objects,
                VERTICES.len(),
            );
            
            Some(snapshot)
        } else {
            None
        }
    }

    #[wasm_bindgen]
    pub fn get_visible_objects(&self) -> u32 {
        self.visible_objects
    }
    
    #[wasm_bindgen]
    pub fn get_total_objects(&self) -> u32 {
        self.total_objects
    }
    
    #[wasm_bindgen]
    pub fn get_culling_ratio(&self) -> f32 {
        if self.total_objects > 0 {
            (self.total_objects - self.visible_objects) as f32 / self.total_objects as f32
        } else {
            0.0
        }
    }

    #[wasm_bindgen]
    pub fn create_test_objects(&mut self, count: u32) {
        self.objects.clear();
        
        // Create a grid of cubes for testing frustum culling
        let grid_size = (count as f32).cbrt().ceil() as i32;
        let spacing = 3.0;
        let offset = (grid_size as f32 - 1.0) * spacing * 0.5;
        
        for x in 0..grid_size {
            for y in 0..grid_size {
                for z in 0..grid_size {
                    if self.objects.len() >= count as usize {
                        break;
                    }
                    
                    let position = Point3::new(
                        x as f32 * spacing - offset,
                        y as f32 * spacing - offset,
                        z as f32 * spacing - offset,
                    );
                    
                    self.objects.push(RenderableObject::new(position, 2.0));
                }
                if self.objects.len() >= count as usize {
                    break;
                }
            }
            if self.objects.len() >= count as usize {
                break;
            }
        }
        
        self.total_objects = self.objects.len() as u32;
        console::log_1(&format!("Created {} test objects for frustum culling", self.total_objects).into());
    }
    
    fn perform_frustum_culling(&mut self) {
        if let Some(ref frustum) = self.current_frustum {
            self.visible_objects = 0;
            
            for object in &mut self.objects {
                object.visible = frustum.contains_sphere(
                    object.bounding_sphere.center,
                    object.bounding_sphere.radius,
                );
                
                if object.visible {
                    self.visible_objects += 1;
                }
            }
        } else {
            // No frustum available, mark all as visible
            self.visible_objects = self.total_objects;
            for object in &mut self.objects {
                object.visible = true;
            }
        }
    }

    fn update_instance_data(&mut self) {
        // Clear previous instance data
        self.instance_data.clear();
        
        // Populate instance data from visible objects only
        for object in &self.objects {
            if object.visible {
                let color = if object.position.x == 0.0 && object.position.y == 0.0 && object.position.z == 0.0 {
                    // Default cube at origin gets beautiful purple-pink color
                    [0.8, 0.2, 0.8]
                } else {
                    // Other objects get height-based coloring
                    match object.position.y {
                        y if y > 2.0 => [1.0, 0.2, 0.2], // Red for high objects
                        y if y < -2.0 => [0.2, 0.2, 1.0], // Blue for low objects
                        _ => [0.2, 1.0, 0.2], // Green for middle objects
                    }
                };
                
                self.instance_data.push(InstanceData::new(
                    [object.position.x, object.position.y, object.position.z],
                    color,
                    object.bounding_sphere.radius,
                ));
            }
        }
        
        // Update instance buffer if we have data
        if !self.instance_data.is_empty() {
            let byte_data = bytemuck::cast_slice(&self.instance_data);
            self.queue.write_buffer(
                &self.instance_buffer,
                0,
                byte_data,
            );
        }
    }

    #[wasm_bindgen]
    pub fn add_object(&mut self, x: f32, y: f32, z: f32, radius: f32) {
        let position = Point3::new(x, y, z);
        let object = RenderableObject::new(position, radius * 2.0); // size = diameter
        self.objects.push(object);
        self.total_objects = self.objects.len() as u32;
        
        // Update cached geometry metrics after scene change
        self.update_geometry_metrics();
    }
    
    #[wasm_bindgen]
    pub fn enable_instancing_demo_with_size(&mut self, grid_size: u32) {
        // Create a grid of colorful cubes to demonstrate instancing
        self.objects.clear();
        
        let grid_size = grid_size as i32;
        
        // Calculate total number of cubes needed
        let total_cubes = (grid_size as u64).pow(3);
        
        // Check if we exceed our buffer capacity
        if total_cubes > self.max_instances as u64 {
            let max_grid_size = (self.max_instances as f64).cbrt().floor() as u32;
            console::log_1(&format!("❌ ERROR: {}x{}x{} grid needs {} cubes, but buffer supports max {} cubes", 
                grid_size, grid_size, grid_size, total_cubes, self.max_instances).into());
            console::log_1(&format!("📏 Maximum supported grid size: {}x{}x{} = {} cubes", 
                max_grid_size, max_grid_size, max_grid_size, max_grid_size.pow(3)).into());
            
            // Fall back to maximum safe grid size
            let safe_grid_size = max_grid_size as i32;
            console::log_1(&format!("🔧 Using safe grid size: {}x{}x{}", safe_grid_size, safe_grid_size, safe_grid_size).into());
            self.enable_instancing_demo_with_size(safe_grid_size as u32);
            return;
        }
        
        // Simple logic: total space is always 1.0 unit
        // For N cubes along an axis: each cube diameter = 1.0/N
        let cube_diameter = 1.0f32 / grid_size as f32;
        let cube_size = cube_diameter / 2.0f32; // radius = diameter / 2
        
        // Spacing between cube centers = cube diameter (touching cubes, no gaps)
        let spacing = cube_diameter + cube_size * 3.0f32;
        
        // Create grid positions centered around origin
        for i in 0..grid_size {
            for j in 0..grid_size {
                for k in 0..grid_size {
                    // Convert grid indices to centered positions
                    let x = if grid_size == 1 {
                        0.0f32 // Single cube at origin
                    } else {
                        (i as f32 - (grid_size as f32 - 1.0f32) / 2.0f32) * spacing
                    };
                    let y = if grid_size == 1 {
                        0.0f32
                    } else {
                        (j as f32 - (grid_size as f32 - 1.0f32) / 2.0f32) * spacing
                    };
                    let z = if grid_size == 1 {
                        0.0f32
                    } else {
                        (k as f32 - (grid_size as f32 - 1.0f32) / 2.0f32) * spacing
                    };
                    
                    let position = Point3::new(x, y, z);
                    self.objects.push(RenderableObject::new(position, cube_size));
                }
            }
        }
        
        self.total_objects = self.objects.len() as u32;
        let total_spread = grid_size as f32 * cube_diameter;
        console::log_1(&format!("🎨 Instancing demo: {}x{}x{} grid = {} cubes (cube_diameter: {:.3}, cube_radius: {:.3}, total_spread: {:.3})", 
            grid_size, grid_size, grid_size, self.total_objects, cube_diameter, cube_size, total_spread).into());
        
        // Update cached geometry metrics after scene change
        self.update_geometry_metrics();
    }
    
    // Update cached geometry metrics - call only when scene changes
    fn update_geometry_metrics(&mut self) {
        self.cached_object_count = self.objects.len() as u32;
        
        // For current cube-based system
        if !self.objects.is_empty() {
            self.cached_vertex_count = (self.objects.len() as u32) * (VERTICES.len() as u32);
            self.cached_index_count = (self.objects.len() as u32) * (INDICES.len() as u32);
            self.cached_edge_count = (self.objects.len() as u32) * 12; // 12 edges per cube
        } else {
            self.cached_vertex_count = 0;
            self.cached_index_count = 0; 
            self.cached_edge_count = 0;
        }
        
        console::log_1(&format!("📐 Geometry metrics updated: {} objects, {} vertices, {} indices, {} edges", 
            self.cached_object_count, self.cached_vertex_count, self.cached_index_count, self.cached_edge_count).into());
    }
    
    // Calculate current GPU memory usage in bytes
    fn calculate_current_memory_usage(&self) -> u64 {
        let vertex_buffer_size = std::mem::size_of_val(VERTICES) as u64;
        let index_buffer_size = std::mem::size_of_val(INDICES) as u64;
        let instance_buffer_size = (std::mem::size_of::<InstanceData>() * self.max_instances as usize) as u64;
        let uniform_buffer_size = std::mem::size_of::<Uniforms>() as u64;
        let depth_texture_size = (self.width * self.height * 4) as u64; // 32-bit depth = 4 bytes per pixel
        
        vertex_buffer_size + index_buffer_size + instance_buffer_size + uniform_buffer_size + depth_texture_size
    }
    
    // Calculate total scene memory (all objects, regardless of visibility)
    fn calculate_scene_size_memory_bytes(&self) -> u64 {
        if self.total_objects == 0 {
            return 0;
        }
        
        // Per-object memory calculation
        let instance_data_per_object = std::mem::size_of::<InstanceData>() as u64;
        
        // The vertex and index buffers are shared across all objects, so we only scale instance data
        let shared_geometry_memory = std::mem::size_of_val(VERTICES) as u64 + std::mem::size_of_val(INDICES) as u64;
        let instance_memory = (self.total_objects as u64) * instance_data_per_object;
        
        shared_geometry_memory + instance_memory
    }
    
    // Calculate active view memory (visible objects only, post-culling)
    fn calculate_active_view_memory_bytes(&self) -> u64 {
        if self.visible_objects == 0 {
            return 0;
        }
        
        // Per-visible-object memory calculation
        let instance_data_per_object = std::mem::size_of::<InstanceData>() as u64;
        
        // The vertex and index buffers are still shared, but we only count visible instance data
        let shared_geometry_memory = if self.visible_objects > 0 { 
            std::mem::size_of_val(VERTICES) as u64 + std::mem::size_of_val(INDICES) as u64 
        } else { 
            0 
        };
        let visible_instance_memory = (self.visible_objects as u64) * instance_data_per_object;
        
        shared_geometry_memory + visible_instance_memory
    }
} 