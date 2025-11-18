/**
 * Device Capability Detection for 3D Performance Optimization
 * Detects device capabilities and provides performance scaling recommendations
 */

export interface DeviceCapabilities {
  gpu: {
    vendor: string;
    renderer: string;
    maxTextureSize: number;
    maxVertexUniforms: number;
    maxFragmentUniforms: number;
    maxVaryingVectors: number;
    maxVertexAttribs: number;
    maxRenderBufferSize: number;
    extensions: string[];
  };
  performance: {
    tier: 'low' | 'medium' | 'high';
    score: number;
    memoryMB: number;
    cores: number;
  };
  display: {
    pixelRatio: number;
    width: number;
    height: number;
    refreshRate: number;
  };
  webgl: {
    version: number;
    supported: boolean;
    contextLost: boolean;
  };
}

export interface PerformanceSettings {
  particleCount: number;
  lodLevels: number;
  shadowQuality: 'off' | 'low' | 'medium' | 'high';
  antialiasing: boolean;
  postProcessing: boolean;
  animationQuality: 'low' | 'medium' | 'high';
  maxFPS: number;
  renderScale: number;
}

class DeviceCapabilityDetector {
  private static instance: DeviceCapabilityDetector;
  private capabilities: DeviceCapabilities | null = null;
  private performanceSettings: PerformanceSettings | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private gl: WebGLRenderingContext | WebGL2RenderingContext | null = null;

  private constructor() {}

  public static getInstance(): DeviceCapabilityDetector {
    if (!DeviceCapabilityDetector.instance) {
      DeviceCapabilityDetector.instance = new DeviceCapabilityDetector();
    }
    return DeviceCapabilityDetector.instance;
  }

  public async detectCapabilities(): Promise<DeviceCapabilities> {
    if (this.capabilities) {
      return this.capabilities;
    }

    this.canvas = document.createElement('canvas');
    this.gl = this.canvas.getContext('webgl2') || this.canvas.getContext('webgl');

    if (!this.gl) {
      throw new Error('WebGL not supported');
    }

    const gpu = this.detectGPUCapabilities();
    const performance = await this.detectPerformanceCapabilities();
    const display = this.detectDisplayCapabilities();
    const webgl = this.detectWebGLCapabilities();

    this.capabilities = {
      gpu,
      performance,
      display,
      webgl
    };

    return this.capabilities;
  }

  private detectGPUCapabilities() {
    if (!this.gl) throw new Error('WebGL context not available');

    const debugInfo = this.gl.getExtension('WEBGL_debug_renderer_info');
    const vendor = debugInfo ? 
      this.gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : 
      this.gl.getParameter(this.gl.VENDOR);
    const renderer = debugInfo ? 
      this.gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : 
      this.gl.getParameter(this.gl.RENDERER);

    const extensions = this.gl.getSupportedExtensions() || [];

    return {
      vendor: vendor as string,
      renderer: renderer as string,
      maxTextureSize: this.gl.getParameter(this.gl.MAX_TEXTURE_SIZE),
      maxVertexUniforms: this.gl.getParameter(this.gl.MAX_VERTEX_UNIFORM_VECTORS),
      maxFragmentUniforms: this.gl.getParameter(this.gl.MAX_FRAGMENT_UNIFORM_VECTORS),
      maxVaryingVectors: this.gl.getParameter(this.gl.MAX_VARYING_VECTORS),
      maxVertexAttribs: this.gl.getParameter(this.gl.MAX_VERTEX_ATTRIBS),
      maxRenderBufferSize: this.gl.getParameter(this.gl.MAX_RENDERBUFFER_SIZE),
      extensions
    };
  }

  private async detectPerformanceCapabilities() {
    const memoryInfo = (navigator as any).deviceMemory || 4; // Default to 4GB if not available
    const cores = navigator.hardwareConcurrency || 4;
    
    // Performance benchmark
    const score = await this.runPerformanceBenchmark();
    
    let tier: 'low' | 'medium' | 'high' = 'medium';
    if (score < 30) tier = 'low';
    else if (score > 70) tier = 'high';

    return {
      tier,
      score,
      memoryMB: memoryInfo * 1024,
      cores
    };
  }

  private detectDisplayCapabilities() {
    return {
      pixelRatio: window.devicePixelRatio || 1,
      width: window.screen.width,
      height: window.screen.height,
      refreshRate: (window.screen as any).refreshRate || 60
    };
  }

  private detectWebGLCapabilities() {
    const isWebGL2 = this.gl instanceof WebGL2RenderingContext;
    
    return {
      version: isWebGL2 ? 2 : 1,
      supported: !!this.gl,
      contextLost: false
    };
  }

  private async runPerformanceBenchmark(): Promise<number> {
    if (!this.gl || !this.canvas) return 50;

    const startTime = performance.now();
    
    // Simple rendering benchmark
    this.canvas.width = 512;
    this.canvas.height = 512;
    
    const vertexShader = this.createShader(this.gl.VERTEX_SHADER, `
      attribute vec2 position;
      void main() {
        gl_Position = vec4(position, 0.0, 1.0);
      }
    `);
    
    const fragmentShader = this.createShader(this.gl.FRAGMENT_SHADER, `
      precision mediump float;
      uniform float time;
      void main() {
        vec2 uv = gl_FragCoord.xy / vec2(512.0);
        float color = sin(uv.x * 10.0 + time) * sin(uv.y * 10.0 + time);
        gl_FragColor = vec4(color, color * 0.5, color * 0.8, 1.0);
      }
    `);

    if (!vertexShader || !fragmentShader) return 50;

    const program = this.gl.createProgram();
    if (!program) return 50;

    this.gl.attachShader(program, vertexShader);
    this.gl.attachShader(program, fragmentShader);
    this.gl.linkProgram(program);

    if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
      return 50;
    }

    // Render frames and measure performance
    let frames = 0;
    const maxFrames = 60;
    
    const renderFrame = () => {
      if (frames >= maxFrames) return;
      
      this.gl!.useProgram(program);
      this.gl!.uniform1f(this.gl!.getUniformLocation(program, 'time'), frames * 0.1);
      this.gl!.drawArrays(this.gl!.TRIANGLES, 0, 6);
      
      frames++;
      requestAnimationFrame(renderFrame);
    };

    return new Promise((resolve) => {
      renderFrame();
      
      setTimeout(() => {
        const endTime = performance.now();
        const duration = endTime - startTime;
        const fps = (frames / duration) * 1000;
        const score = Math.min(100, (fps / 60) * 100);
        resolve(score);
      }, 1000);
    });
  }

  private createShader(type: number, source: string): WebGLShader | null {
    if (!this.gl) return null;
    
    const shader = this.gl.createShader(type);
    if (!shader) return null;
    
    this.gl.shaderSource(shader, source);
    this.gl.compileShader(shader);
    
    if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
      this.gl.deleteShader(shader);
      return null;
    }
    
    return shader;
  }

  public getOptimalSettings(): PerformanceSettings {
    if (this.performanceSettings) {
      return this.performanceSettings;
    }

    if (!this.capabilities) {
      throw new Error('Capabilities not detected. Call detectCapabilities() first.');
    }

    const { performance, display, gpu } = this.capabilities;
    
    let settings: PerformanceSettings;

    switch (performance.tier) {
      case 'low':
        settings = {
          particleCount: 50,
          lodLevels: 2,
          shadowQuality: 'off',
          antialiasing: false,
          postProcessing: false,
          animationQuality: 'low',
          maxFPS: 30,
          renderScale: 0.75
        };
        break;
      
      case 'medium':
        settings = {
          particleCount: 200,
          lodLevels: 3,
          shadowQuality: 'low',
          antialiasing: display.pixelRatio <= 1,
          postProcessing: false,
          animationQuality: 'medium',
          maxFPS: 60,
          renderScale: 1.0
        };
        break;
      
      case 'high':
        settings = {
          particleCount: 500,
          lodLevels: 4,
          shadowQuality: 'high',
          antialiasing: true,
          postProcessing: true,
          animationQuality: 'high',
          maxFPS: display.refreshRate || 60,
          renderScale: Math.min(display.pixelRatio, 2.0)
        };
        break;
    }

    // Adjust based on memory constraints
    if (performance.memoryMB < 2048) {
      settings.particleCount = Math.floor(settings.particleCount * 0.5);
      settings.renderScale = Math.min(settings.renderScale, 0.8);
    }

    // Adjust based on GPU capabilities
    if (gpu.maxTextureSize < 4096) {
      settings.shadowQuality = 'off';
      settings.postProcessing = false;
    }

    this.performanceSettings = settings;
    return settings;
  }

  public getCapabilities(): DeviceCapabilities | null {
    return this.capabilities;
  }

  public cleanup(): void {
    if (this.canvas) {
      this.canvas.remove();
      this.canvas = null;
    }
    this.gl = null;
  }
}

export default DeviceCapabilityDetector;