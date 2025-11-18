/**
 * Efficient Particle System Manager for Cyberpunk Effects
 * Manages particle pools, batching, and performance optimization
 */

import * as THREE from 'three';

export interface ParticleSystemConfig {
  maxParticles: number;
  particleSize: number;
  lifetime: number;
  emissionRate: number;
  velocity: THREE.Vector3;
  acceleration: THREE.Vector3;
  color: THREE.Color;
  opacity: number;
  blending: THREE.Blending;
  texture?: THREE.Texture;
}

export interface ParticleEmitter {
  id: string;
  position: THREE.Vector3;
  config: ParticleSystemConfig;
  active: boolean;
  lastEmissionTime: number;
  particleCount: number;
}

interface Particle {
  position: THREE.Vector3;
  velocity: THREE.Vector3;
  life: number;
  maxLife: number;
  size: number;
  color: THREE.Color;
  opacity: number;
  active: boolean;
}

export interface ParticlePool {
  particles: Particle[];
  geometry: THREE.BufferGeometry;
  material: THREE.PointsMaterial;
  points: THREE.Points;
  activeCount: number;
}

class ParticleSystemManager {
  private static instance: ParticleSystemManager;
  private emitters: Map<string, ParticleEmitter> = new Map();
  private pools: Map<string, ParticlePool> = new Map();
  private scene: THREE.Scene | null = null;
  private maxTotalParticles: number = 10000;
  private currentParticleCount: number = 0;
  private performanceTier: 'low' | 'medium' | 'high' = 'medium';
  private frameSkipCounter: number = 0;
  private updateFrequency: number = 1; // Update every N frames

  private constructor() {}

  public static getInstance(): ParticleSystemManager {
    if (!ParticleSystemManager.instance) {
      ParticleSystemManager.instance = new ParticleSystemManager();
    }
    return ParticleSystemManager.instance;
  }

  public setScene(scene: THREE.Scene): void {
    this.scene = scene;
  }

  public setPerformanceTier(tier: 'low' | 'medium' | 'high'): void {
    this.performanceTier = tier;
    
    // Adjust settings based on performance tier
    switch (tier) {
      case 'low':
        this.maxTotalParticles = 2000;
        this.updateFrequency = 3; // Update every 3 frames
        break;
      case 'medium':
        this.maxTotalParticles = 5000;
        this.updateFrequency = 2; // Update every 2 frames
        break;
      case 'high':
        this.maxTotalParticles = 15000;
        this.updateFrequency = 1; // Update every frame
        break;
    }
  }

  public createParticlePool(
    poolId: string,
    config: ParticleSystemConfig
  ): ParticlePool {
    // Adjust particle count based on performance tier
    const adjustedMaxParticles = this.getAdjustedParticleCount(config.maxParticles);
    
    const particles: Particle[] = [];
    const positions = new Float32Array(adjustedMaxParticles * 3);
    const colors = new Float32Array(adjustedMaxParticles * 3);
    const sizes = new Float32Array(adjustedMaxParticles);
    const opacities = new Float32Array(adjustedMaxParticles);

    // Initialize particle pool
    for (let i = 0; i < adjustedMaxParticles; i++) {
      particles.push({
        position: new THREE.Vector3(),
        velocity: new THREE.Vector3(),
        life: 0,
        maxLife: config.lifetime,
        size: config.particleSize,
        color: config.color.clone(),
        opacity: 0,
        active: false
      });

      // Initialize buffer arrays
      positions[i * 3] = 0;
      positions[i * 3 + 1] = 0;
      positions[i * 3 + 2] = 0;

      colors[i * 3] = config.color.r;
      colors[i * 3 + 1] = config.color.g;
      colors[i * 3 + 2] = config.color.b;

      sizes[i] = 0;
      opacities[i] = 0;
    }

    // Create geometry and material
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
    geometry.setAttribute('opacity', new THREE.BufferAttribute(opacities, 1));

    const material = new THREE.PointsMaterial({
      size: config.particleSize,
      vertexColors: true,
      transparent: true,
      opacity: config.opacity,
      blending: config.blending,
      map: config.texture,
      sizeAttenuation: true,
      alphaTest: 0.01
    });

    // Enable custom shader attributes for opacity
    material.onBeforeCompile = (shader) => {
      shader.vertexShader = shader.vertexShader.replace(
        'attribute float size;',
        'attribute float size;\nattribute float opacity;'
      );
      shader.vertexShader = shader.vertexShader.replace(
        'void main() {',
        'varying float vOpacity;\nvoid main() {\nvOpacity = opacity;'
      );
      
      shader.fragmentShader = shader.fragmentShader.replace(
        'void main() {',
        'varying float vOpacity;\nvoid main() {'
      );
      shader.fragmentShader = shader.fragmentShader.replace(
        'gl_FragColor = vec4( diffuse, opacity * totalOpacity );',
        'gl_FragColor = vec4( diffuse, opacity * totalOpacity * vOpacity );'
      );
    };

    const points = new THREE.Points(geometry, material);
    points.frustumCulled = false; // Disable frustum culling for better performance

    const pool: ParticlePool = {
      particles,
      geometry,
      material,
      points,
      activeCount: 0
    };

    this.pools.set(poolId, pool);

    if (this.scene) {
      this.scene.add(points);
    }

    return pool;
  }

  public createEmitter(
    emitterId: string,
    position: THREE.Vector3,
    config: ParticleSystemConfig,
    poolId?: string
  ): ParticleEmitter {
    // Create pool if it doesn't exist
    if (!poolId) {
      poolId = `${emitterId}_pool`;
    }
    
    if (!this.pools.has(poolId)) {
      this.createParticlePool(poolId, config);
    }

    const emitter: ParticleEmitter = {
      id: emitterId,
      position: position.clone(),
      config,
      active: true,
      lastEmissionTime: 0,
      particleCount: 0
    };

    this.emitters.set(emitterId, emitter);
    return emitter;
  }

  public updateEmitter(emitterId: string, deltaTime: number): void {
    const emitter = this.emitters.get(emitterId);
    if (!emitter || !emitter.active) return;

    const poolId = `${emitterId}_pool`;
    const pool = this.pools.get(poolId);
    if (!pool) return;

    const currentTime = performance.now();
    const timeSinceLastEmission = currentTime - emitter.lastEmissionTime;
    const emissionInterval = 1000 / emitter.config.emissionRate; // ms between emissions

    // Emit new particles
    if (timeSinceLastEmission >= emissionInterval && 
        this.currentParticleCount < this.maxTotalParticles) {
      this.emitParticle(emitter, pool);
      emitter.lastEmissionTime = currentTime;
    }

    // Update existing particles (with frame skipping for performance)
    if (this.frameSkipCounter % this.updateFrequency === 0) {
      this.updateParticles(pool, deltaTime);
    }
  }

  private emitParticle(emitter: ParticleEmitter, pool: ParticlePool): void {
    // Find inactive particle
    const particle = pool.particles.find(p => !p.active);
    if (!particle) return;

    // Initialize particle
    particle.position.copy(emitter.position);
    particle.velocity.copy(emitter.config.velocity);
    particle.velocity.add(new THREE.Vector3(
      (Math.random() - 0.5) * 0.1,
      (Math.random() - 0.5) * 0.1,
      (Math.random() - 0.5) * 0.1
    ));
    particle.life = particle.maxLife;
    particle.size = emitter.config.particleSize * (0.8 + Math.random() * 0.4);
    particle.color.copy(emitter.config.color);
    particle.opacity = emitter.config.opacity;
    particle.active = true;

    pool.activeCount++;
    this.currentParticleCount++;
    emitter.particleCount++;
  }

  private updateParticles(pool: ParticlePool, deltaTime: number): void {
    const positions = pool.geometry.attributes.position.array as Float32Array;
    const colors = pool.geometry.attributes.color.array as Float32Array;
    const sizes = pool.geometry.attributes.size.array as Float32Array;
    const opacities = pool.geometry.attributes.opacity.array as Float32Array;

    let activeCount = 0;

    pool.particles.forEach((particle, index) => {
      if (!particle.active) {
        // Hide inactive particles
        sizes[index] = 0;
        opacities[index] = 0;
        return;
      }

      // Update particle physics
      particle.velocity.add(
        new THREE.Vector3(0, -9.8, 0).multiplyScalar(deltaTime * 0.001)
      );
      particle.position.add(
        particle.velocity.clone().multiplyScalar(deltaTime * 0.001)
      );
      particle.life -= deltaTime;

      // Update visual properties
      const lifeRatio = particle.life / particle.maxLife;
      particle.opacity = particle.opacity * lifeRatio;
      particle.size = particle.size * (0.5 + lifeRatio * 0.5);

      // Update buffer arrays
      positions[index * 3] = particle.position.x;
      positions[index * 3 + 1] = particle.position.y;
      positions[index * 3 + 2] = particle.position.z;

      colors[index * 3] = particle.color.r;
      colors[index * 3 + 1] = particle.color.g;
      colors[index * 3 + 2] = particle.color.b;

      sizes[index] = particle.size;
      opacities[index] = particle.opacity;

      // Check if particle should die
      if (particle.life <= 0 || particle.opacity < 0.01) {
        particle.active = false;
        pool.activeCount--;
        this.currentParticleCount--;
        sizes[index] = 0;
        opacities[index] = 0;
      } else {
        activeCount++;
      }
    });

    // Mark attributes as needing update
    pool.geometry.attributes.position.needsUpdate = true;
    pool.geometry.attributes.color.needsUpdate = true;
    pool.geometry.attributes.size.needsUpdate = true;
    pool.geometry.attributes.opacity.needsUpdate = true;

    pool.activeCount = activeCount;
  }

  public update(deltaTime: number): void {
    this.frameSkipCounter++;
    
    this.emitters.forEach((emitter, emitterId) => {
      this.updateEmitter(emitterId, deltaTime);
    });
  }

  public setEmitterActive(emitterId: string, active: boolean): void {
    const emitter = this.emitters.get(emitterId);
    if (emitter) {
      emitter.active = active;
    }
  }

  public setEmitterPosition(emitterId: string, position: THREE.Vector3): void {
    const emitter = this.emitters.get(emitterId);
    if (emitter) {
      emitter.position.copy(position);
    }
  }

  public removeEmitter(emitterId: string): void {
    const emitter = this.emitters.get(emitterId);
    if (!emitter) return;

    // Remove from scene
    const poolId = `${emitterId}_pool`;
    const pool = this.pools.get(poolId);
    if (pool && this.scene) {
      this.scene.remove(pool.points);
      
      // Dispose of resources
      pool.geometry.dispose();
      pool.material.dispose();
      if (pool.material.map) {
        pool.material.map.dispose();
      }
    }

    this.emitters.delete(emitterId);
    this.pools.delete(poolId);
  }

  private getAdjustedParticleCount(baseCount: number): number {
    const multiplier = this.performanceTier === 'low' ? 0.3 : 
                     this.performanceTier === 'medium' ? 0.6 : 1.0;
    return Math.floor(baseCount * multiplier);
  }

  public getStats(): {
    totalEmitters: number;
    totalParticles: number;
    activeParticles: number;
    performanceTier: string;
  } {
    let activeParticles = 0;
    this.pools.forEach(pool => {
      activeParticles += pool.activeCount;
    });

    return {
      totalEmitters: this.emitters.size,
      totalParticles: this.maxTotalParticles,
      activeParticles,
      performanceTier: this.performanceTier
    };
  }

  public cleanup(): void {
    this.emitters.forEach((emitter, emitterId) => {
      this.removeEmitter(emitterId);
    });
    
    this.emitters.clear();
    this.pools.clear();
    this.scene = null;
    this.currentParticleCount = 0;
  }
}

export default ParticleSystemManager;