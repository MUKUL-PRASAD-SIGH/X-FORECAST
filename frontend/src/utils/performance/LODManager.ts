/**
 * Level of Detail (LOD) Manager for 3D Visualizations
 * Dynamically adjusts rendering complexity based on distance and performance
 */

import * as THREE from 'three';

export interface LODLevel {
  distance: number;
  particleCount: number;
  geometryComplexity: number;
  textureResolution: number;
  animationFrameSkip: number;
  enableEffects: boolean;
}

export interface LODConfiguration {
  levels: LODLevel[];
  transitionDistance: number;
  hysteresis: number; // Prevents flickering between levels
}

export interface GeometryLOD {
  high: THREE.BufferGeometry;
  medium: THREE.BufferGeometry;
  low: THREE.BufferGeometry;
}

class LODManager {
  private static instance: LODManager;
  private camera: THREE.Camera | null = null;
  private lodObjects: Map<string, THREE.LOD> = new Map();
  private configurations: Map<string, LODConfiguration> = new Map();
  private frameCount: number = 0;
  private lastUpdateTime: number = 0;
  private updateInterval: number = 100; // Update LOD every 100ms

  private constructor() {}

  public static getInstance(): LODManager {
    if (!LODManager.instance) {
      LODManager.instance = new LODManager();
    }
    return LODManager.instance;
  }

  public setCamera(camera: THREE.Camera): void {
    this.camera = camera;
  }

  public createLODConfiguration(
    objectType: string,
    performanceTier: 'low' | 'medium' | 'high'
  ): LODConfiguration {
    let config: LODConfiguration;

    switch (performanceTier) {
      case 'low':
        config = {
          levels: [
            {
              distance: 0,
              particleCount: 25,
              geometryComplexity: 0.3,
              textureResolution: 256,
              animationFrameSkip: 2,
              enableEffects: false
            },
            {
              distance: 10,
              particleCount: 10,
              geometryComplexity: 0.1,
              textureResolution: 128,
              animationFrameSkip: 4,
              enableEffects: false
            }
          ],
          transitionDistance: 2,
          hysteresis: 0.5
        };
        break;

      case 'medium':
        config = {
          levels: [
            {
              distance: 0,
              particleCount: 100,
              geometryComplexity: 0.7,
              textureResolution: 512,
              animationFrameSkip: 1,
              enableEffects: true
            },
            {
              distance: 15,
              particleCount: 50,
              geometryComplexity: 0.4,
              textureResolution: 256,
              animationFrameSkip: 2,
              enableEffects: false
            },
            {
              distance: 30,
              particleCount: 20,
              geometryComplexity: 0.2,
              textureResolution: 128,
              animationFrameSkip: 3,
              enableEffects: false
            }
          ],
          transitionDistance: 3,
          hysteresis: 1.0
        };
        break;

      case 'high':
        config = {
          levels: [
            {
              distance: 0,
              particleCount: 300,
              geometryComplexity: 1.0,
              textureResolution: 1024,
              animationFrameSkip: 0,
              enableEffects: true
            },
            {
              distance: 20,
              particleCount: 150,
              geometryComplexity: 0.7,
              textureResolution: 512,
              animationFrameSkip: 1,
              enableEffects: true
            },
            {
              distance: 40,
              particleCount: 75,
              geometryComplexity: 0.4,
              textureResolution: 256,
              animationFrameSkip: 2,
              enableEffects: false
            },
            {
              distance: 60,
              particleCount: 25,
              geometryComplexity: 0.2,
              textureResolution: 128,
              animationFrameSkip: 3,
              enableEffects: false
            }
          ],
          transitionDistance: 5,
          hysteresis: 2.0
        };
        break;
    }

    this.configurations.set(objectType, config);
    return config;
  }

  public createGeometryLOD(
    baseGeometry: THREE.BufferGeometry,
    complexityLevels: number[] = [1.0, 0.6, 0.3]
  ): GeometryLOD {
    const geometries: { [key: string]: THREE.BufferGeometry } = {};
    
    // High detail (original)
    geometries.high = baseGeometry.clone();
    
    // Medium detail
    geometries.medium = this.simplifyGeometry(baseGeometry, complexityLevels[1]);
    
    // Low detail
    geometries.low = this.simplifyGeometry(baseGeometry, complexityLevels[2]);

    return {
      high: geometries.high,
      medium: geometries.medium,
      low: geometries.low
    };
  }

  private simplifyGeometry(
    geometry: THREE.BufferGeometry,
    complexity: number
  ): THREE.BufferGeometry {
    const simplified = geometry.clone();
    
    // Simple vertex reduction by sampling
    const positionAttribute = simplified.getAttribute('position');
    if (positionAttribute) {
      const originalCount = positionAttribute.count;
      const targetCount = Math.floor(originalCount * complexity);
      
      if (targetCount < originalCount) {
        const step = Math.floor(originalCount / targetCount);
        const newPositions = [];
        const newIndices = [];
        
        for (let i = 0; i < originalCount; i += step) {
          newPositions.push(
            positionAttribute.getX(i),
            positionAttribute.getY(i),
            positionAttribute.getZ(i)
          );
        }
        
        // Update geometry
        simplified.setAttribute('position', new THREE.Float32BufferAttribute(newPositions, 3));
        
        // Update indices if they exist
        if (simplified.index) {
          for (let i = 0; i < newPositions.length / 3 - 2; i++) {
            newIndices.push(i, i + 1, i + 2);
          }
          simplified.setIndex(newIndices);
        }
      }
    }
    
    return simplified;
  }

  public createLODObject(
    objectId: string,
    geometries: GeometryLOD,
    materials: { high: THREE.Material; medium: THREE.Material; low: THREE.Material },
    objectType: string = 'default'
  ): THREE.LOD {
    const lod = new THREE.LOD();
    const config = this.configurations.get(objectType);
    
    if (!config) {
      throw new Error(`LOD configuration not found for object type: ${objectType}`);
    }

    // Add LOD levels
    const highMesh = new THREE.Mesh(geometries.high, materials.high);
    const mediumMesh = new THREE.Mesh(geometries.medium, materials.medium);
    const lowMesh = new THREE.Mesh(geometries.low, materials.low);

    lod.addLevel(highMesh, 0);
    lod.addLevel(mediumMesh, config.levels[1]?.distance || 15);
    lod.addLevel(lowMesh, config.levels[2]?.distance || 30);

    this.lodObjects.set(objectId, lod);
    return lod;
  }

  public updateLOD(deltaTime: number): void {
    if (!this.camera) return;

    this.frameCount++;
    const currentTime = performance.now();
    
    // Throttle updates for performance
    if (currentTime - this.lastUpdateTime < this.updateInterval) {
      return;
    }
    
    this.lastUpdateTime = currentTime;

    this.lodObjects.forEach((lod, objectId) => {
      lod.update(this.camera!);
    });
  }

  public getCurrentLODLevel(objectId: string): number {
    const lod = this.lodObjects.get(objectId);
    if (!lod || !this.camera) return 0;

    const distance = lod.position.distanceTo(this.camera.position);
    const config = this.getConfigurationForObject(objectId);
    
    if (!config) return 0;

    for (let i = config.levels.length - 1; i >= 0; i--) {
      if (distance >= config.levels[i].distance) {
        return i;
      }
    }
    
    return 0;
  }

  public getLODSettings(objectId: string): LODLevel | null {
    const levelIndex = this.getCurrentLODLevel(objectId);
    const config = this.getConfigurationForObject(objectId);
    
    if (!config || levelIndex >= config.levels.length) return null;
    
    return config.levels[levelIndex];
  }

  private getConfigurationForObject(objectId: string): LODConfiguration | null {
    // Try to find configuration by object type (could be enhanced with object metadata)
    const entries = Array.from(this.configurations.entries());
    for (const [type, config] of entries) {
      if (objectId.includes(type)) {
        return config;
      }
    }
    
    // Return default configuration if available
    return this.configurations.get('default') || null;
  }

  public shouldSkipFrame(objectId: string): boolean {
    const settings = this.getLODSettings(objectId);
    if (!settings || settings.animationFrameSkip === 0) return false;
    
    return this.frameCount % (settings.animationFrameSkip + 1) !== 0;
  }

  public getParticleCount(objectId: string, baseCount: number): number {
    const settings = this.getLODSettings(objectId);
    if (!settings) return baseCount;
    
    return Math.min(baseCount, settings.particleCount);
  }

  public shouldEnableEffects(objectId: string): boolean {
    const settings = this.getLODSettings(objectId);
    return settings?.enableEffects ?? true;
  }

  public getTextureResolution(objectId: string, baseResolution: number): number {
    const settings = this.getLODSettings(objectId);
    if (!settings) return baseResolution;
    
    return Math.min(baseResolution, settings.textureResolution);
  }

  public getGeometryComplexity(objectId: string): number {
    const settings = this.getLODSettings(objectId);
    return settings?.geometryComplexity ?? 1.0;
  }

  public removeLODObject(objectId: string): void {
    const lod = this.lodObjects.get(objectId);
    if (lod) {
      // Clean up geometries and materials
      lod.levels.forEach(level => {
        const mesh = level.object as THREE.Mesh;
        if (mesh.geometry) mesh.geometry.dispose();
        if (mesh.material) {
          if (Array.isArray(mesh.material)) {
            mesh.material.forEach(mat => mat.dispose());
          } else {
            mesh.material.dispose();
          }
        }
      });
      
      this.lodObjects.delete(objectId);
    }
  }

  public cleanup(): void {
    this.lodObjects.forEach((lod, objectId) => {
      this.removeLODObject(objectId);
    });
    this.lodObjects.clear();
    this.configurations.clear();
    this.camera = null;
  }

  public getStats(): {
    objectCount: number;
    averageLODLevel: number;
    frameCount: number;
  } {
    let totalLODLevel = 0;
    let objectCount = 0;

    this.lodObjects.forEach((lod, objectId) => {
      totalLODLevel += this.getCurrentLODLevel(objectId);
      objectCount++;
    });

    return {
      objectCount,
      averageLODLevel: objectCount > 0 ? totalLODLevel / objectCount : 0,
      frameCount: this.frameCount
    };
  }
}

export default LODManager;