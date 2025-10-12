import React, { useRef, useEffect, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Line } from '@react-three/drei';
import * as THREE from 'three';
import styled from 'styled-components';

// Particle system configurations
interface ParticleConfig {
  count: number;
  size: number;
  color: string;
  speed: number;
  behavior: 'floating' | 'streaming' | 'reactive' | 'orbital';
  opacity: number;
}

// Advanced particle system with multiple behaviors
export const AdvancedParticleSystem: React.FC<ParticleConfig> = ({
  count,
  size,
  color,
  speed,
  behavior,
  opacity
}) => {
  const meshRef = useRef<THREE.Points>(null);
  const velocitiesRef = useRef<Float32Array>();
  
  const { positions, colors, velocities } = useMemo(() => {
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const velocities = new Float32Array(count * 3);
    
    const colorObj = new THREE.Color(color);
    
    for (let i = 0; i < count; i++) {
      // Initial positions
      positions[i * 3] = (Math.random() - 0.5) * 20;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 20;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 20;
      
      // Colors with slight variation
      colors[i * 3] = colorObj.r + (Math.random() - 0.5) * 0.2;
      colors[i * 3 + 1] = colorObj.g + (Math.random() - 0.5) * 0.2;
      colors[i * 3 + 2] = colorObj.b + (Math.random() - 0.5) * 0.2;
      
      // Initial velocities
      velocities[i * 3] = (Math.random() - 0.5) * speed;
      velocities[i * 3 + 1] = (Math.random() - 0.5) * speed;
      velocities[i * 3 + 2] = (Math.random() - 0.5) * speed;
    }
    
    velocitiesRef.current = velocities;
    return { positions, colors, velocities };
  }, [count, color, speed]);
  
  useFrame((state) => {
    if (meshRef.current && velocitiesRef.current) {
      const positions = meshRef.current.geometry.attributes.position.array as Float32Array;
      const time = state.clock.elapsedTime;
      
      for (let i = 0; i < count; i++) {
        const i3 = i * 3;
        
        switch (behavior) {
          case 'floating':
            positions[i3] += Math.sin(time + i) * 0.01;
            positions[i3 + 1] += Math.cos(time + i) * 0.01;
            positions[i3 + 2] += Math.sin(time * 0.5 + i) * 0.005;
            break;
            
          case 'streaming':
            positions[i3] += velocitiesRef.current[i3] * 0.1;
            positions[i3 + 1] += velocitiesRef.current[i3 + 1] * 0.1;
            positions[i3 + 2] += velocitiesRef.current[i3 + 2] * 0.1;
            
            // Reset particles that go too far
            if (Math.abs(positions[i3]) > 15) {
              positions[i3] = (Math.random() - 0.5) * 20;
              positions[i3 + 1] = (Math.random() - 0.5) * 20;
              positions[i3 + 2] = (Math.random() - 0.5) * 20;
            }
            break;
            
          case 'orbital':
            const radius = 5 + Math.sin(time + i) * 2;
            positions[i3] = Math.cos(time * speed + i) * radius;
            positions[i3 + 1] = Math.sin(time * speed + i) * radius;
            positions[i3 + 2] = Math.sin(time * 0.5 + i) * 2;
            break;
            
          case 'reactive':
            // React to mouse position (simplified)
            const mouseInfluence = 0.01;
            positions[i3] += (Math.random() - 0.5) * mouseInfluence;
            positions[i3 + 1] += (Math.random() - 0.5) * mouseInfluence;
            break;
        }
      }
      
      meshRef.current.geometry.attributes.position.needsUpdate = true;
    }
  });
  
  return (
    <points ref={meshRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={count}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={count}
          array={colors}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={size}
        vertexColors
        transparent
        opacity={opacity}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
};

// Holographic shader material
const holographicVertexShader = `
  varying vec2 vUv;
  varying vec3 vPosition;
  uniform float time;
  
  void main() {
    vUv = uv;
    vPosition = position;
    
    vec3 pos = position;
    pos.y += sin(pos.x * 10.0 + time) * 0.1;
    pos.x += cos(pos.z * 8.0 + time) * 0.05;
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
  }
`;

const holographicFragmentShader = `
  varying vec2 vUv;
  varying vec3 vPosition;
  uniform float time;
  uniform vec3 color;
  uniform float opacity;
  
  void main() {
    vec2 uv = vUv;
    
    // Scanline effect
    float scanline = sin(uv.y * 800.0 + time * 10.0) * 0.04;
    
    // Holographic interference
    float interference = sin(uv.x * 100.0 + time * 5.0) * sin(uv.y * 100.0 + time * 3.0) * 0.1;
    
    // Edge glow
    float edge = 1.0 - smoothstep(0.0, 0.1, min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y)));
    
    vec3 finalColor = color + scanline + interference + edge * 0.5;
    float finalOpacity = opacity * (0.7 + scanline + interference);
    
    gl_FragColor = vec4(finalColor, finalOpacity);
  }
`;

// Holographic material component
export const HolographicMaterial: React.FC<{
  color: string;
  opacity: number;
  animated?: boolean;
}> = ({ color, opacity, animated = true }) => {
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  
  useFrame((state) => {
    if (materialRef.current && animated) {
      materialRef.current.uniforms.time.value = state.clock.elapsedTime;
    }
  });
  
  const uniforms = useMemo(() => ({
    time: { value: 0 },
    color: { value: new THREE.Color(color) },
    opacity: { value: opacity }
  }), [color, opacity]);
  
  return (
    <shaderMaterial
      ref={materialRef}
      uniforms={uniforms}
      vertexShader={holographicVertexShader}
      fragmentShader={holographicFragmentShader}
      transparent
      side={THREE.DoubleSide}
    />
  );
};

// Glitch effect component
export const GlitchEffect: React.FC<{
  intensity: number;
  speed: number;
  children: React.ReactNode;
}> = ({ intensity, speed, children }) => {
  const groupRef = useRef<THREE.Group>(null);
  
  useFrame((state) => {
    if (groupRef.current) {
      const time = state.clock.elapsedTime;
      const glitchX = Math.sin(time * speed * 10) * intensity * 0.1;
      const glitchY = Math.cos(time * speed * 15) * intensity * 0.05;
      
      groupRef.current.position.x = glitchX;
      groupRef.current.position.y = glitchY;
      
      // Random glitch spikes
      if (Math.random() < 0.01 * intensity) {
        groupRef.current.position.x += (Math.random() - 0.5) * intensity;
        groupRef.current.position.y += (Math.random() - 0.5) * intensity;
      }
    }
  });
  
  return <group ref={groupRef}>{children}</group>;
};

// Energy field effect
export const EnergyField: React.FC<{
  radius: number;
  color: string;
  intensity: number;
}> = ({ radius, color, intensity }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.01;
      meshRef.current.rotation.z += 0.005;
      
      const scale = 1 + Math.sin(state.clock.elapsedTime * 2) * 0.1 * intensity;
      meshRef.current.scale.setScalar(scale);
    }
  });
  
  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[radius, 32, 32]} />
      <meshBasicMaterial
        color={color}
        transparent
        opacity={0.1 * intensity}
        wireframe
      />
    </mesh>
  );
};

// Data stream effect
export const DataStream: React.FC<{
  points: THREE.Vector3[];
  color: string;
  speed: number;
}> = ({ points, color, speed }) => {
  const lineRef = useRef<THREE.Line>(null);
  const particlesRef = useRef<THREE.Points>(null);
  
  const streamParticles = useMemo(() => {
    const particleCount = 50;
    const positions = new Float32Array(particleCount * 3);
    
    for (let i = 0; i < particleCount; i++) {
      const t = i / particleCount;
      const pointIndex = Math.floor(t * (points.length - 1));
      const point = points[pointIndex];
      
      positions[i * 3] = point.x;
      positions[i * 3 + 1] = point.y;
      positions[i * 3 + 2] = point.z;
    }
    
    return positions;
  }, [points]);
  
  useFrame((state) => {
    if (particlesRef.current) {
      const positions = particlesRef.current.geometry.attributes.position.array as Float32Array;
      const time = state.clock.elapsedTime * speed;
      
      for (let i = 0; i < 50; i++) {
        const t = (i / 50 + time) % 1;
        const pointIndex = Math.floor(t * (points.length - 1));
        const nextIndex = Math.min(pointIndex + 1, points.length - 1);
        const localT = t * (points.length - 1) - pointIndex;
        
        const point = points[pointIndex];
        const nextPoint = points[nextIndex];
        
        positions[i * 3] = point.x + (nextPoint.x - point.x) * localT;
        positions[i * 3 + 1] = point.y + (nextPoint.y - point.y) * localT;
        positions[i * 3 + 2] = point.z + (nextPoint.z - point.z) * localT;
      }
      
      particlesRef.current.geometry.attributes.position.needsUpdate = true;
    }
  });
  
  return (
    <group>
      {/* Stream line */}
      <Line
        points={points.map(p => [p.x, p.y, p.z])}
        color={color}
        lineWidth={2}
        transparent
        opacity={0.3}
      />
      
      {/* Moving particles */}
      <points ref={particlesRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={50}
            array={streamParticles}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial
          color={color}
          size={0.1}
          transparent
          opacity={0.8}
          sizeAttenuation
        />
      </points>
    </group>
  );
};

// CSS-based cyberpunk effects for 2D elements
export const CyberpunkEffectsCSS = styled.div`
  /* Glitch text effect */
  .glitch-text {
    position: relative;
    color: ${props => props.theme.colors.primaryText};
    
    &::before,
    &::after {
      content: attr(data-text);
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
    }
    
    &::before {
      animation: glitch-1 0.5s infinite;
      color: ${props => props.theme.colors.neonBlue};
      z-index: -1;
    }
    
    &::after {
      animation: glitch-2 0.5s infinite;
      color: ${props => props.theme.colors.hotPink};
      z-index: -2;
    }
  }
  
  @keyframes glitch-1 {
    0%, 14%, 15%, 49%, 50%, 99%, 100% {
      transform: translate(0);
    }
    15%, 49% {
      transform: translate(-2px, 1px);
    }
  }
  
  @keyframes glitch-2 {
    0%, 20%, 21%, 62%, 63%, 99%, 100% {
      transform: translate(0);
    }
    21%, 62% {
      transform: translate(2px, -1px);
    }
  }
  
  /* Scanline effect */
  .scanlines {
    position: relative;
    overflow: hidden;
    
    &::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: linear-gradient(
        transparent 50%,
        rgba(0, 255, 255, 0.03) 50%
      );
      background-size: 100% 4px;
      pointer-events: none;
      animation: scanlines 0.1s linear infinite;
    }
  }
  
  @keyframes scanlines {
    0% { transform: translateY(0); }
    100% { transform: translateY(4px); }
  }
  
  /* Holographic border effect */
  .holographic-border {
    position: relative;
    border: 1px solid transparent;
    background: linear-gradient(45deg, 
      ${props => props.theme.colors.neonBlue}, 
      ${props => props.theme.colors.hotPink}, 
      ${props => props.theme.colors.acidGreen}
    ) border-box;
    -webkit-mask: linear-gradient(#fff 0 0) padding-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: destination-out;
    mask: linear-gradient(#fff 0 0) padding-box, linear-gradient(#fff 0 0);
    mask-composite: exclude;
    
    &::before {
      content: '';
      position: absolute;
      top: -2px;
      left: -2px;
      right: -2px;
      bottom: -2px;
      background: linear-gradient(45deg, 
        ${props => props.theme.colors.neonBlue}, 
        ${props => props.theme.colors.hotPink}, 
        ${props => props.theme.colors.acidGreen}
      );
      border-radius: inherit;
      z-index: -1;
      animation: holographic-rotate 3s linear infinite;
    }
  }
  
  @keyframes holographic-rotate {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
  
  /* Matrix rain effect */
  .matrix-rain {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: -1;
    opacity: 0.1;
    
    &::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background-image: 
        linear-gradient(transparent 70%, ${props => props.theme.colors.acidGreen} 70%);
      background-size: 20px 20px;
      animation: matrix-fall 2s linear infinite;
    }
  }
  
  @keyframes matrix-fall {
    0% { transform: translateY(-100%); }
    100% { transform: translateY(100vh); }
  }
`;

// Cyberpunk loading spinner
export const CyberpunkSpinner: React.FC<{
  size?: number;
  color?: string;
  speed?: number;
}> = ({ size = 50, color = '#00FFFF', speed = 1 }) => {
  return (
    <div
      style={{
        width: size,
        height: size,
        border: `2px solid transparent`,
        borderTop: `2px solid ${color}`,
        borderRadius: '50%',
        animation: `spin ${1 / speed}s linear infinite`,
        filter: `drop-shadow(0 0 10px ${color})`,
      }}
    />
  );
};

// Export all effects
export default {
  AdvancedParticleSystem,
  HolographicMaterial,
  GlitchEffect,
  EnergyField,
  DataStream,
  CyberpunkEffectsCSS,
  CyberpunkSpinner
};