/**
 * WebGL Shader Performance Optimizer for Cyberpunk Effects
 * Provides optimized shaders and performance monitoring
 */

import * as THREE from 'three';

export interface ShaderPerformanceMetrics {
  compileTime: number;
  drawCalls: number;
  trianglesRendered: number;
  textureBinds: number;
  shaderSwitches: number;
}

export interface OptimizedShaderConfig {
  vertexShader: string;
  fragmentShader: string;
  uniforms: { [key: string]: THREE.IUniform };
  defines: { [key: string]: any };
  performanceLevel: 'low' | 'medium' | 'high';
}

class WebGLShaderOptimizer {
  private static instance: WebGLShaderOptimizer;
  private shaderCache: Map<string, THREE.ShaderMaterial> = new Map();
  private performanceMetrics: ShaderPerformanceMetrics = {
    compileTime: 0,
    drawCalls: 0,
    trianglesRendered: 0,
    textureBinds: 0,
    shaderSwitches: 0
  };
  private performanceLevel: 'low' | 'medium' | 'high' = 'medium';

  private constructor() {}

  public static getInstance(): WebGLShaderOptimizer {
    if (!WebGLShaderOptimizer.instance) {
      WebGLShaderOptimizer.instance = new WebGLShaderOptimizer();
    }
    return WebGLShaderOptimizer.instance;
  }

  public setPerformanceLevel(level: 'low' | 'medium' | 'high'): void {
    this.performanceLevel = level;
  }

  public getOptimizedCyberpunkShader(type: 'holographic' | 'particle' | 'glow' | 'scan'): OptimizedShaderConfig {
    switch (type) {
      case 'holographic':
        return this.getHolographicShader();
      case 'particle':
        return this.getParticleShader();
      case 'glow':
        return this.getGlowShader();
      case 'scan':
        return this.getScanShader();
      default:
        throw new Error(`Unknown shader type: ${type}`);
    }
  }

  private getHolographicShader(): OptimizedShaderConfig {
    const defines: { [key: string]: any } = {};
    
    // Performance-based defines
    if (this.performanceLevel === 'high') {
      defines.HIGH_QUALITY = true;
      defines.ENABLE_REFLECTIONS = true;
      defines.ENABLE_DISTORTION = true;
    } else if (this.performanceLevel === 'medium') {
      defines.MEDIUM_QUALITY = true;
      defines.ENABLE_DISTORTION = true;
    } else {
      defines.LOW_QUALITY = true;
    }

    const vertexShader = `
      uniform float time;
      uniform float intensity;
      varying vec2 vUv;
      varying vec3 vPosition;
      varying vec3 vNormal;
      
      ${this.performanceLevel === 'high' ? `
        varying vec3 vWorldPosition;
        varying vec3 vViewPosition;
      ` : ''}
      
      void main() {
        vUv = uv;
        vPosition = position;
        vNormal = normal;
        
        vec3 pos = position;
        
        #ifdef HIGH_QUALITY
          // High-quality vertex displacement
          pos += normal * sin(time * 2.0 + position.x * 5.0) * 0.02 * intensity;
          pos += normal * cos(time * 1.5 + position.y * 3.0) * 0.015 * intensity;
          vWorldPosition = (modelMatrix * vec4(pos, 1.0)).xyz;
          vViewPosition = (modelViewMatrix * vec4(pos, 1.0)).xyz;
        #elif defined(MEDIUM_QUALITY)
          // Medium-quality vertex displacement
          pos += normal * sin(time + position.x * 3.0) * 0.015 * intensity;
          vWorldPosition = (modelMatrix * vec4(pos, 1.0)).xyz;
        #else
          // Low-quality - minimal displacement
          pos += normal * sin(time + position.x) * 0.01 * intensity;
        #endif
        
        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
      }
    `;

    const fragmentShader = `
      uniform float time;
      uniform float intensity;
      uniform vec3 color;
      uniform float opacity;
      uniform sampler2D noiseTexture;
      
      varying vec2 vUv;
      varying vec3 vPosition;
      varying vec3 vNormal;
      
      ${this.performanceLevel === 'high' ? `
        varying vec3 vWorldPosition;
        varying vec3 vViewPosition;
      ` : ''}
      
      // Optimized noise function
      float noise(vec2 uv) {
        #ifdef HIGH_QUALITY
          return texture2D(noiseTexture, uv * 4.0 + time * 0.1).r;
        #else
          // Simplified noise for better performance
          return fract(sin(dot(uv, vec2(12.9898, 78.233))) * 43758.5453);
        #endif
      }
      
      void main() {
        vec2 uv = vUv;
        
        #ifdef ENABLE_DISTORTION
          // UV distortion for holographic effect
          float distortion = noise(uv * 2.0) * 0.02 * intensity;
          uv += vec2(distortion, distortion * 0.5);
        #endif
        
        // Base holographic color
        vec3 holo = color;
        
        #ifdef HIGH_QUALITY
          // High-quality holographic effects
          float fresnel = pow(1.0 - dot(normalize(vViewPosition), vNormal), 2.0);
          float scanline = sin(uv.y * 800.0 + time * 10.0) * 0.1 + 0.9;
          float flicker = sin(time * 30.0) * 0.02 + 0.98;
          
          holo *= fresnel * scanline * flicker;
          holo += vec3(0.0, 0.3, 0.6) * fresnel * 0.5;
          
        #elif defined(MEDIUM_QUALITY)
          // Medium-quality effects
          float scanline = sin(uv.y * 400.0 + time * 5.0) * 0.05 + 0.95;
          float flicker = sin(time * 15.0) * 0.01 + 0.99;
          
          holo *= scanline * flicker;
          
        #else
          // Low-quality - minimal effects
          float scanline = sin(uv.y * 200.0 + time * 2.0) * 0.03 + 0.97;
          holo *= scanline;
        #endif
        
        // Edge glow
        float edge = 1.0 - abs(dot(vNormal, vec3(0.0, 0.0, 1.0)));
        holo += color * edge * 0.3 * intensity;
        
        gl_FragColor = vec4(holo, opacity * intensity);
      }
    `;

    return {
      vertexShader,
      fragmentShader,
      uniforms: {
        time: { value: 0.0 },
        intensity: { value: 1.0 },
        color: { value: new THREE.Color(0x00ffff) },
        opacity: { value: 0.8 },
        noiseTexture: { value: this.createNoiseTexture() }
      },
      defines,
      performanceLevel: this.performanceLevel
    };
  }

  private getParticleShader(): OptimizedShaderConfig {
    const defines: { [key: string]: any } = {};
    
    if (this.performanceLevel === 'high') {
      defines.HIGH_QUALITY = true;
      defines.ENABLE_ROTATION = true;
    } else if (this.performanceLevel === 'medium') {
      defines.MEDIUM_QUALITY = true;
    } else {
      defines.LOW_QUALITY = true;
    }

    const vertexShader = `
      uniform float time;
      uniform float size;
      attribute float particleSize;
      attribute float particleOpacity;
      attribute vec3 particleColor;
      
      ${this.performanceLevel === 'high' ? 'attribute float rotation;' : ''}
      
      varying float vOpacity;
      varying vec3 vColor;
      ${this.performanceLevel === 'high' ? 'varying float vRotation;' : ''}
      
      void main() {
        vOpacity = particleOpacity;
        vColor = particleColor;
        
        #ifdef ENABLE_ROTATION
          vRotation = rotation;
        #endif
        
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        
        #ifdef HIGH_QUALITY
          // High-quality size calculation with distance attenuation
          float distance = length(mvPosition.xyz);
          gl_PointSize = particleSize * size * (300.0 / distance);
        #else
          // Simplified size calculation
          gl_PointSize = particleSize * size;
        #endif
        
        gl_Position = projectionMatrix * mvPosition;
      }
    `;

    const fragmentShader = `
      uniform float time;
      uniform sampler2D particleTexture;
      
      varying float vOpacity;
      varying vec3 vColor;
      ${this.performanceLevel === 'high' ? 'varying float vRotation;' : ''}
      
      void main() {
        vec2 uv = gl_PointCoord;
        
        #ifdef ENABLE_ROTATION
          // Rotate UV coordinates
          float c = cos(vRotation);
          float s = sin(vRotation);
          uv = vec2(
            c * (uv.x - 0.5) - s * (uv.y - 0.5) + 0.5,
            s * (uv.x - 0.5) + c * (uv.y - 0.5) + 0.5
          );
        #endif
        
        #ifdef HIGH_QUALITY
          // High-quality particle with texture and glow
          vec4 texColor = texture2D(particleTexture, uv);
          float glow = 1.0 - length(uv - 0.5) * 2.0;
          glow = pow(glow, 2.0);
          
          vec3 finalColor = vColor * texColor.rgb + vColor * glow * 0.5;
          float finalOpacity = vOpacity * texColor.a * glow;
          
        #elif defined(MEDIUM_QUALITY)
          // Medium-quality particle
          float dist = length(uv - 0.5);
          float alpha = 1.0 - smoothstep(0.0, 0.5, dist);
          
          vec3 finalColor = vColor;
          float finalOpacity = vOpacity * alpha;
          
        #else
          // Low-quality - simple circular particle
          float dist = length(uv - 0.5);
          float alpha = step(dist, 0.5);
          
          vec3 finalColor = vColor;
          float finalOpacity = vOpacity * alpha;
        #endif
        
        gl_FragColor = vec4(finalColor, finalOpacity);
      }
    `;

    return {
      vertexShader,
      fragmentShader,
      uniforms: {
        time: { value: 0.0 },
        size: { value: 1.0 },
        particleTexture: { value: this.createParticleTexture() }
      },
      defines,
      performanceLevel: this.performanceLevel
    };
  }

  private getGlowShader(): OptimizedShaderConfig {
    const defines: { [key: string]: any } = {};
    
    if (this.performanceLevel === 'high') {
      defines.HIGH_QUALITY = true;
      defines.ENABLE_BLOOM = true;
    } else if (this.performanceLevel === 'medium') {
      defines.MEDIUM_QUALITY = true;
    } else {
      defines.LOW_QUALITY = true;
    }

    const vertexShader = `
      varying vec2 vUv;
      varying vec3 vNormal;
      varying vec3 vPosition;
      
      void main() {
        vUv = uv;
        vNormal = normalize(normalMatrix * normal);
        vPosition = position;
        
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `;

    const fragmentShader = `
      uniform float time;
      uniform vec3 glowColor;
      uniform float intensity;
      uniform float power;
      
      varying vec2 vUv;
      varying vec3 vNormal;
      varying vec3 vPosition;
      
      void main() {
        #ifdef HIGH_QUALITY
          // High-quality glow with multiple layers
          float glow1 = pow(1.0 - abs(dot(vNormal, vec3(0.0, 0.0, 1.0))), power);
          float glow2 = pow(1.0 - abs(dot(vNormal, vec3(0.0, 0.0, 1.0))), power * 0.5);
          float pulse = sin(time * 3.0) * 0.1 + 0.9;
          
          vec3 finalColor = glowColor * (glow1 + glow2 * 0.3) * intensity * pulse;
          
        #elif defined(MEDIUM_QUALITY)
          // Medium-quality glow
          float glow = pow(1.0 - abs(dot(vNormal, vec3(0.0, 0.0, 1.0))), power);
          float pulse = sin(time * 2.0) * 0.05 + 0.95;
          
          vec3 finalColor = glowColor * glow * intensity * pulse;
          
        #else
          // Low-quality - simple glow
          float glow = 1.0 - abs(dot(vNormal, vec3(0.0, 0.0, 1.0)));
          vec3 finalColor = glowColor * glow * intensity;
        #endif
        
        gl_FragColor = vec4(finalColor, glow * intensity);
      }
    `;

    return {
      vertexShader,
      fragmentShader,
      uniforms: {
        time: { value: 0.0 },
        glowColor: { value: new THREE.Color(0x00ffff) },
        intensity: { value: 1.0 },
        power: { value: 2.0 }
      },
      defines,
      performanceLevel: this.performanceLevel
    };
  }

  private getScanShader(): OptimizedShaderConfig {
    const defines: { [key: string]: any } = {};
    
    if (this.performanceLevel === 'high') {
      defines.HIGH_QUALITY = true;
      defines.ENABLE_INTERFERENCE = true;
    } else if (this.performanceLevel === 'medium') {
      defines.MEDIUM_QUALITY = true;
    } else {
      defines.LOW_QUALITY = true;
    }

    const vertexShader = `
      varying vec2 vUv;
      
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `;

    const fragmentShader = `
      uniform float time;
      uniform vec3 scanColor;
      uniform float scanSpeed;
      uniform float scanWidth;
      uniform float intensity;
      
      varying vec2 vUv;
      
      void main() {
        vec2 uv = vUv;
        
        #ifdef HIGH_QUALITY
          // High-quality scan with interference
          float scan = sin((uv.y + time * scanSpeed) * 50.0) * 0.5 + 0.5;
          scan = pow(scan, scanWidth);
          
          float interference = sin(uv.x * 100.0 + time * 10.0) * 0.1;
          scan += interference;
          
          float flicker = sin(time * 25.0) * 0.02 + 0.98;
          scan *= flicker;
          
        #elif defined(MEDIUM_QUALITY)
          // Medium-quality scan
          float scan = sin((uv.y + time * scanSpeed) * 30.0) * 0.5 + 0.5;
          scan = pow(scan, scanWidth);
          
          float flicker = sin(time * 15.0) * 0.01 + 0.99;
          scan *= flicker;
          
        #else
          // Low-quality - simple scan
          float scan = sin((uv.y + time * scanSpeed) * 20.0) * 0.5 + 0.5;
          scan = pow(scan, scanWidth);
        #endif
        
        vec3 finalColor = scanColor * scan * intensity;
        gl_FragColor = vec4(finalColor, scan * intensity);
      }
    `;

    return {
      vertexShader,
      fragmentShader,
      uniforms: {
        time: { value: 0.0 },
        scanColor: { value: new THREE.Color(0x00ff00) },
        scanSpeed: { value: 1.0 },
        scanWidth: { value: 2.0 },
        intensity: { value: 1.0 }
      },
      defines,
      performanceLevel: this.performanceLevel
    };
  }

  public createOptimizedMaterial(shaderConfig: OptimizedShaderConfig): THREE.ShaderMaterial {
    const cacheKey = `${shaderConfig.performanceLevel}_${JSON.stringify(shaderConfig.defines)}`;
    
    if (this.shaderCache.has(cacheKey)) {
      return this.shaderCache.get(cacheKey)!.clone();
    }

    const startTime = performance.now();
    
    const material = new THREE.ShaderMaterial({
      vertexShader: shaderConfig.vertexShader,
      fragmentShader: shaderConfig.fragmentShader,
      uniforms: shaderConfig.uniforms,
      defines: shaderConfig.defines,
      transparent: true,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide
    });

    const compileTime = performance.now() - startTime;
    this.performanceMetrics.compileTime += compileTime;

    this.shaderCache.set(cacheKey, material);
    return material.clone();
  }

  private createNoiseTexture(): THREE.Texture {
    const size = this.performanceLevel === 'high' ? 256 : 
                 this.performanceLevel === 'medium' ? 128 : 64;
    
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    
    const context = canvas.getContext('2d')!;
    const imageData = context.createImageData(size, size);
    
    for (let i = 0; i < imageData.data.length; i += 4) {
      const noise = Math.random();
      imageData.data[i] = noise * 255;     // R
      imageData.data[i + 1] = noise * 255; // G
      imageData.data[i + 2] = noise * 255; // B
      imageData.data[i + 3] = 255;         // A
    }
    
    context.putImageData(imageData, 0, 0);
    
    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    
    return texture;
  }

  private createParticleTexture(): THREE.Texture {
    const size = this.performanceLevel === 'high' ? 64 : 32;
    
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    
    const context = canvas.getContext('2d')!;
    const gradient = context.createRadialGradient(
      size / 2, size / 2, 0,
      size / 2, size / 2, size / 2
    );
    
    gradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
    gradient.addColorStop(0.5, 'rgba(255, 255, 255, 0.5)');
    gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
    
    context.fillStyle = gradient;
    context.fillRect(0, 0, size, size);
    
    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    
    return texture;
  }

  public updateMetrics(drawCalls: number, triangles: number): void {
    this.performanceMetrics.drawCalls += drawCalls;
    this.performanceMetrics.trianglesRendered += triangles;
  }

  public getPerformanceMetrics(): ShaderPerformanceMetrics {
    return { ...this.performanceMetrics };
  }

  public resetMetrics(): void {
    this.performanceMetrics = {
      compileTime: 0,
      drawCalls: 0,
      trianglesRendered: 0,
      textureBinds: 0,
      shaderSwitches: 0
    };
  }

  public cleanup(): void {
    this.shaderCache.forEach(material => {
      material.dispose();
    });
    this.shaderCache.clear();
  }
}

export default WebGLShaderOptimizer;