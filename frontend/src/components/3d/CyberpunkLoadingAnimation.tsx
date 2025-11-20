import React, { useRef, useEffect, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Text, Sphere, Box, Torus, Cylinder } from '@react-three/drei';
import * as THREE from 'three';
import { motion, AnimatePresence } from 'framer-motion';
import styled from 'styled-components';

interface ModelTrainingStatus {
  model: string;
  status: 'pending' | 'training' | 'completed' | 'failed';
  progress: number;
  eta?: string;
  accuracy?: number;
}

interface CyberpunkLoadingAnimationProps {
  modelStatuses: ModelTrainingStatus[];
  overallProgress: number;
  isTraining: boolean;
  trainingMessage?: string;
  width?: number;
  height?: number;
  showParticles?: boolean;
  animationIntensity?: number;
}

const LoadingContainer = styled(motion.div).withConfig({
  shouldForwardProp: (prop) => !['$width', '$height', '$isTraining'].includes(prop)
})<{ $width: number; $height: number; $isTraining?: boolean }>`
  width: ${props => props.$width}px;
  height: ${props => props.$height}px;
  position: relative;
  background: 
    radial-gradient(circle at 50% 50%, rgba(0, 255, 255, 0.1) 0%, transparent 70%),
    linear-gradient(45deg, rgba(255, 20, 147, 0.05) 0%, rgba(57, 255, 20, 0.05) 100%);
  border: 2px solid transparent;
  border-radius: 12px;
  overflow: hidden;
  
  &::before {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    background: linear-gradient(45deg, 
      rgba(0, 255, 255, 1) 0%, 
      rgba(255, 20, 147, 1) 25%,
      rgba(57, 255, 20, 1) 50%,
      rgba(255, 20, 147, 1) 75%,
      rgba(0, 255, 255, 1) 100%
    );
    border-radius: 12px;
    z-index: -1;
    animation: ${props => props.$isTraining ? 'trainingBorderGlow 1s ease-in-out infinite' : 'none'};
  }
  
  @keyframes trainingBorderGlow {
    0%, 100% { opacity: 0.8; filter: blur(2px); }
    50% { opacity: 1; filter: blur(0px); }
  }
`;

const StatusOverlay = styled.div`
  position: absolute;
  top: 15px;
  left: 15px;
  right: 15px;
  z-index: 10;
  background: rgba(0, 0, 0, 0.9);
  border: 1px solid rgba(0, 255, 255, 0.5);
  border-radius: 6px;
  padding: 12px;
  font-family: 'Courier New', monospace;
  backdrop-filter: blur(4px);
`;

const StatusHeader = styled.div`
  color: #00ffff;
  font-size: 14px;
  font-weight: bold;
  margin-bottom: 8px;
  text-align: center;
  text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
`;

const ProgressBar = styled.div.withConfig({
  shouldForwardProp: (prop) => !['$progress'].includes(prop)
})<{ $progress: number }>`
  width: 100%;
  height: 6px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 3px;
  margin: 8px 0;
  overflow: hidden;
  position: relative;
  
  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    width: ${props => props.$progress}%;
    background: linear-gradient(90deg, #00ffff 0%, #ff1493 50%, #39ff14 100%);
    border-radius: 3px;
    transition: width 0.3s ease;
    animation: progressGlow 2s ease-in-out infinite;
  }
  
  @keyframes progressGlow {
    0%, 100% { box-shadow: 0 0 5px rgba(0, 255, 255, 0.5); }
    50% { box-shadow: 0 0 15px rgba(0, 255, 255, 0.8); }
  }
`;

const ModelStatusGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 8px;
  margin-top: 10px;
`;

const ModelStatusCard = styled.div.withConfig({
  shouldForwardProp: (prop) => !['$status'].includes(prop)
})<{ $status: string }>`
  background: rgba(0, 0, 0, 0.7);
  border: 1px solid ${props => {
    switch (props.$status) {
      case 'completed': return '#39ff14';
      case 'training': return '#00ffff';
      case 'failed': return '#ff6b6b';
      default: return 'rgba(255, 255, 255, 0.3)';
    }
  }};
  border-radius: 4px;
  padding: 6px;
  font-size: 10px;
  text-align: center;
  color: ${props => {
    switch (props.$status) {
      case 'completed': return '#39ff14';
      case 'training': return '#00ffff';
      case 'failed': return '#ff6b6b';
      default: return 'rgba(255, 255, 255, 0.7)';
    }
  }};
  
  ${props => props.$status === 'training' && `
    animation: cardPulse 1.5s ease-in-out infinite;
  `}
  
  @keyframes cardPulse {
    0%, 100% { 
      box-shadow: 0 0 5px ${props => {
        switch (props.$status) {
          case 'training': return 'rgba(0, 255, 255, 0.3)';
          default: return 'transparent';
        }
      }};
    }
    50% { 
      box-shadow: 0 0 15px ${props => {
        switch (props.$status) {
          case 'training': return 'rgba(0, 255, 255, 0.6)';
          default: return 'transparent';
        }
      }};
    }
  }
`;

// Rotating Neural Network Visualization
const NeuralNetworkVisualization: React.FC<{
  progress: number;
  isTraining: boolean;
  intensity: number;
}> = ({ progress, isTraining, intensity }) => {
  const networkRef = useRef<THREE.Group>(null);
  const nodesRef = useRef<THREE.Group>(null);
  
  useFrame((state) => {
    if (networkRef.current && isTraining) {
      const time = state.clock.elapsedTime;
      networkRef.current.rotation.y = time * 0.5 * intensity;
      networkRef.current.rotation.x = Math.sin(time * 0.3) * 0.1;
    }
    
    if (nodesRef.current && isTraining) {
      const time = state.clock.elapsedTime;
      nodesRef.current.children.forEach((child, index) => {
        if (child instanceof THREE.Mesh) {
          child.rotation.y = time * (1 + index * 0.1) * intensity;
          const scale = 1 + Math.sin(time * 2 + index) * 0.2;
          child.scale.setScalar(scale);
        }
      });
    }
  });
  
  // Create neural network nodes
  const layers = [
    { count: 4, y: 1.5, color: '#ff6b6b' },
    { count: 6, y: 0.5, color: '#4ecdc4' },
    { count: 6, y: -0.5, color: '#45b7d1' },
    { count: 3, y: -1.5, color: '#39ff14' }
  ];
  
  return (
    <group ref={networkRef}>
      <group ref={nodesRef}>
        {layers.map((layer, layerIndex) => (
          <group key={layerIndex}>
            {Array.from({ length: layer.count }, (_, nodeIndex) => {
              const angle = (nodeIndex / layer.count) * Math.PI * 2;
              const radius = 0.8 + layerIndex * 0.2;
              const x = Math.cos(angle) * radius;
              const z = Math.sin(angle) * radius;
              
              return (
                <Sphere
                  key={`${layerIndex}-${nodeIndex}`}
                  position={[x, layer.y, z]}
                  args={[0.08, 8, 8]}
                >
                  <meshBasicMaterial
                    color={layer.color}
                    transparent
                    opacity={0.8}
                  />
                </Sphere>
              );
            })}
          </group>
        ))}
      </group>
      
      {/* Connection lines between layers */}
      {layers.slice(0, -1).map((layer, layerIndex) => {
        const nextLayer = layers[layerIndex + 1];
        const connections = [];
        
        for (let i = 0; i < layer.count; i++) {
          for (let j = 0; j < nextLayer.count; j++) {
            const angle1 = (i / layer.count) * Math.PI * 2;
            const radius1 = 0.8 + layerIndex * 0.2;
            const x1 = Math.cos(angle1) * radius1;
            const z1 = Math.sin(angle1) * radius1;
            
            const angle2 = (j / nextLayer.count) * Math.PI * 2;
            const radius2 = 0.8 + (layerIndex + 1) * 0.2;
            const x2 = Math.cos(angle2) * radius2;
            const z2 = Math.sin(angle2) * radius2;
            
            connections.push([
              [x1, layer.y, z1],
              [x2, nextLayer.y, z2]
            ]);
          }
        }
        
        return connections.map((connection, connectionIndex) => (
          <line key={`${layerIndex}-${connectionIndex}`}>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                count={2}
                array={new Float32Array([
                  ...connection[0],
                  ...connection[1]
                ])}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial
              color="#ffffff"
              transparent
              opacity={0.2}
            />
          </line>
        ));
      })}
    </group>
  );
};

// Data Flow Particles
const DataFlowParticles: React.FC<{
  enabled: boolean;
  intensity: number;
}> = ({ enabled, intensity }) => {
  const particlesRef = useRef<THREE.Points>(null);
  const [particles] = useState(() => {
    const count = Math.floor(intensity * 200) + 50;
    const positions = new Float32Array(count * 3);
    const velocities = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const sizes = new Float32Array(count);
    
    for (let i = 0; i < count; i++) {
      // Random positions in a sphere
      const radius = Math.random() * 3;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.random() * Math.PI;
      
      positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = radius * Math.cos(phi);
      
      // Random velocities
      velocities[i * 3] = (Math.random() - 0.5) * 0.02;
      velocities[i * 3 + 1] = (Math.random() - 0.5) * 0.02;
      velocities[i * 3 + 2] = (Math.random() - 0.5) * 0.02;
      
      // Cyberpunk colors
      const colorChoice = Math.random();
      if (colorChoice < 0.33) {
        colors[i * 3] = 0; colors[i * 3 + 1] = 1; colors[i * 3 + 2] = 1; // Cyan
      } else if (colorChoice < 0.66) {
        colors[i * 3] = 1; colors[i * 3 + 1] = 0.08; colors[i * 3 + 2] = 0.58; // Magenta
      } else {
        colors[i * 3] = 0.22; colors[i * 3 + 1] = 1; colors[i * 3 + 2] = 0.08; // Green
      }
      
      sizes[i] = Math.random() * 0.05 + 0.02;
    }
    
    return { positions, velocities, colors, sizes, count };
  });
  
  useFrame((state) => {
    if (!enabled || !particlesRef.current) return;
    
    const time = state.clock.elapsedTime;
    const positionAttribute = particlesRef.current.geometry.attributes.position;
    const positions = positionAttribute.array as Float32Array;
    
    for (let i = 0; i < particles.count; i++) {
      const i3 = i * 3;
      
      // Update positions
      positions[i3] += particles.velocities[i3];
      positions[i3 + 1] += particles.velocities[i3 + 1];
      positions[i3 + 2] += particles.velocities[i3 + 2];
      
      // Add orbital motion
      positions[i3] += Math.sin(time + i * 0.01) * 0.002;
      positions[i3 + 1] += Math.cos(time + i * 0.01) * 0.002;
      
      // Reset particles that drift too far
      const distance = Math.sqrt(
        positions[i3] * positions[i3] +
        positions[i3 + 1] * positions[i3 + 1] +
        positions[i3 + 2] * positions[i3 + 2]
      );
      
      if (distance > 4) {
        const radius = Math.random() * 3;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.random() * Math.PI;
        
        positions[i3] = radius * Math.sin(phi) * Math.cos(theta);
        positions[i3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
        positions[i3 + 2] = radius * Math.cos(phi);
      }
    }
    
    positionAttribute.needsUpdate = true;
  });
  
  if (!enabled) return null;
  
  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={particles.count}
          array={particles.positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={particles.count}
          array={particles.colors}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-size"
          count={particles.count}
          array={particles.sizes}
          itemSize={1}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.03}
        vertexColors
        transparent
        opacity={0.8}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
};

// Holographic Progress Ring
const HolographicProgressRing: React.FC<{
  progress: number;
  isTraining: boolean;
}> = ({ progress, isTraining }) => {
  const ringRef = useRef<THREE.Mesh>(null);
  const progressRingRef = useRef<THREE.Mesh>(null);
  
  useFrame((state) => {
    if (ringRef.current && isTraining) {
      const time = state.clock.elapsedTime;
      ringRef.current.rotation.z = time * 0.5;
    }
    
    if (progressRingRef.current) {
      const time = state.clock.elapsedTime;
      const material = progressRingRef.current.material as THREE.MeshBasicMaterial;
      material.opacity = 0.6 + Math.sin(time * 3) * 0.2;
    }
  });
  
  const progressAngle = (progress / 100) * Math.PI * 2;
  
  return (
    <group position={[0, 0, 0]}>
      {/* Base ring */}
      <mesh ref={ringRef}>
        <ringGeometry args={[2.8, 3, 64]} />
        <meshBasicMaterial
          color="#00ffff"
          transparent
          opacity={0.2}
          side={THREE.DoubleSide}
        />
      </mesh>
      
      {/* Progress ring */}
      <mesh ref={progressRingRef}>
        <ringGeometry args={[2.8, 3, 64, 1, 0, progressAngle]} />
        <meshBasicMaterial
          color="#ff1493"
          transparent
          opacity={0.8}
          side={THREE.DoubleSide}
        />
      </mesh>
      
      {/* Progress text */}
      <Text
        position={[0, 0, 0.1]}
        fontSize={0.3}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
      >
        {progress.toFixed(1)}%
      </Text>
    </group>
  );
};

// Cyberpunk Lighting
const CyberpunkTrainingLighting: React.FC = () => {
  return (
    <>
      <ambientLight intensity={0.4} color="#0a0a2e" />
      <pointLight position={[3, 3, 3]} intensity={1} color="#00ffff" />
      <pointLight position={[-3, -3, -3]} intensity={0.8} color="#ff1493" />
      <pointLight position={[0, 5, -3]} intensity={0.6} color="#39ff14" />
      <spotLight
        position={[0, 8, 0]}
        angle={0.5}
        penumbra={1}
        intensity={0.8}
        color="#00ffff"
        castShadow
      />
    </>
  );
};

// Main Component
export const CyberpunkLoadingAnimation: React.FC<CyberpunkLoadingAnimationProps> = ({
  modelStatuses,
  overallProgress,
  isTraining,
  trainingMessage = "Training Ensemble Models...",
  width = 800,
  height = 600,
  showParticles = true,
  animationIntensity = 1
}) => {
  return (
    <LoadingContainer
      $width={width}
      $height={height}
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      $isTraining={isTraining}
    >
      <StatusOverlay>
        <StatusHeader>{trainingMessage}</StatusHeader>
        <ProgressBar $progress={overallProgress} />
        <div style={{ textAlign: 'center', fontSize: '12px', color: '#ffffff', marginTop: '5px' }}>
          Overall Progress: {overallProgress.toFixed(1)}%
        </div>
        
        <ModelStatusGrid>
          {modelStatuses.map((model) => (
            <ModelStatusCard key={model.model} $status={model.status}>
              <div style={{ fontWeight: 'bold', marginBottom: '2px' }}>
                {model.model.toUpperCase()}
              </div>
              <div style={{ fontSize: '9px', opacity: 0.8 }}>
                {model.status === 'training' && `${model.progress.toFixed(0)}%`}
                {model.status === 'completed' && model.accuracy && `${(model.accuracy * 100).toFixed(1)}%`}
                {model.status === 'failed' && 'ERROR'}
                {model.status === 'pending' && 'WAITING'}
              </div>
              {model.eta && model.status === 'training' && (
                <div style={{ fontSize: '8px', opacity: 0.6, marginTop: '2px' }}>
                  ETA: {model.eta}
                </div>
              )}
            </ModelStatusCard>
          ))}
        </ModelStatusGrid>
      </StatusOverlay>
      
      <Canvas
        camera={{ position: [5, 3, 5], fov: 60 }}
        style={{ background: 'transparent' }}
      >
        <CyberpunkTrainingLighting />
        
        {/* Neural Network Visualization */}
        <NeuralNetworkVisualization
          progress={overallProgress}
          isTraining={isTraining}
          intensity={animationIntensity}
        />
        
        {/* Data Flow Particles */}
        {showParticles && (
          <DataFlowParticles
            enabled={isTraining}
            intensity={animationIntensity}
          />
        )}
        
        {/* Holographic Progress Ring */}
        <HolographicProgressRing
          progress={overallProgress}
          isTraining={isTraining}
        />
        
        {/* Training Status Text */}
        <Text
          position={[0, -3.5, 0]}
          fontSize={0.15}
          color="#00ffff"
          anchorX="center"
          anchorY="middle"
        >
          ENSEMBLE MODEL TRAINING
        </Text>
        
        {isTraining && (
          <Text
            position={[0, -4, 0]}
            fontSize={0.1}
            color="#39ff14"
            anchorX="center"
            anchorY="middle"
          >
            NEURAL NETWORKS OPTIMIZING...
          </Text>
        )}
      </Canvas>
    </LoadingContainer>
  );
};

export default CyberpunkLoadingAnimation;