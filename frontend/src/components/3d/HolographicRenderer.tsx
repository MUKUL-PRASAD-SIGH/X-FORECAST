import React, { useRef, useEffect, useState, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Line, Sphere, Box } from '@react-three/drei';
import * as THREE from 'three';
import { motion } from 'framer-motion';
import styled from 'styled-components';

// Types for 3D data visualization
export interface TimeSeriesData {
  date: string;
  value: number;
  category?: string;
}

export interface CustomerJourney {
  customerId: string;
  touchpoints: Array<{
    timestamp: string;
    event: string;
    value: number;
    position: [number, number, number];
  }>;
}

export interface HolographicRendererProps {
  data: TimeSeriesData[] | CustomerJourney;
  type: 'timeSeries' | 'customerJourney' | 'scatter3d' | 'network';
  theme: 'cyberpunk' | 'neon' | 'hologram';
  interactive?: boolean;
  showParticles?: boolean;
  autoRotate?: boolean;
}

// Styled container for 3D canvas
const CanvasContainer = styled(motion.div)`
  width: 100%;
  height: 400px;
  border: 1px solid ${props => props.theme.colors.neonBlue};
  border-radius: 8px;
  background: ${props => props.theme.colors.darkerBg};
  position: relative;
  overflow: hidden;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: 
      radial-gradient(circle at 50% 50%, rgba(0, 255, 255, 0.1) 0%, transparent 70%);
    pointer-events: none;
    z-index: 1;
  }
`;

const ControlsOverlay = styled.div`
  position: absolute;
  top: 10px;
  right: 10px;
  z-index: 10;
  display: flex;
  gap: 8px;
`;

const ControlButton = styled.button`
  background: ${props => props.theme.colors.cardBg};
  border: 1px solid ${props => props.theme.colors.neonBlue};
  color: ${props => props.theme.colors.primaryText};
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    border-color: ${props => props.theme.colors.hotPink};
    box-shadow: ${props => props.theme.effects.softGlow};
  }
`;

// Particle system component
const ParticleField: React.FC<{ count: number; color: string }> = ({ count, color }) => {
  const meshRef = useRef<THREE.Points>(null);
  
  const particles = useMemo(() => {
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    
    for (let i = 0; i < count; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 20;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 20;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 20;
      
      const colorObj = new THREE.Color(color);
      colors[i * 3] = colorObj.r;
      colors[i * 3 + 1] = colorObj.g;
      colors[i * 3 + 2] = colorObj.b;
    }
    
    return { positions, colors };
  }, [count, color]);
  
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.001;
      meshRef.current.rotation.x += 0.0005;
    }
  });
  
  return (
    <points ref={meshRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={count}
          array={particles.positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={count}
          array={particles.colors}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.05}
        vertexColors
        transparent
        opacity={0.6}
        sizeAttenuation
      />
    </points>
  );
};

// Holographic time series chart
const HolographicTimeSeries: React.FC<{ data: TimeSeriesData[] }> = ({ data }) => {
  const meshRef = useRef<THREE.Group>(null);
  
  const points = useMemo(() => {
    return data.map((point, index) => {
      const x = (index / data.length) * 10 - 5;
      const y = (point.value / Math.max(...data.map(d => d.value))) * 5 - 2.5;
      const z = 0;
      return new THREE.Vector3(x, y, z);
    });
  }, [data]);
  
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
    }
  });
  
  return (
    <group ref={meshRef}>
      {/* Main line */}
      <Line
        points={points}
        color="#00FFFF"
        lineWidth={3}
        transparent
        opacity={0.8}
      />
      
      {/* Data points */}
      {points.map((point, index) => (
        <Sphere
          key={index}
          position={[point.x, point.y, point.z]}
          args={[0.1]}
        >
          <meshBasicMaterial
            color="#FF1493"
            transparent
            opacity={0.7}
          />
        </Sphere>
      ))}
      
      {/* Holographic effect planes */}
      {points.map((point, index) => (
        <mesh
          key={`plane-${index}`}
          position={[point.x, point.y, point.z]}
          rotation={[0, 0, 0]}
        >
          <planeGeometry args={[0.3, 0.3]} />
          <meshBasicMaterial
            color="#00FFFF"
            transparent
            opacity={0.2}
            side={THREE.DoubleSide}
          />
        </mesh>
      ))}
      
      {/* Grid base */}
      <gridHelper
        args={[10, 20, "#39FF14", "#39FF14"]}
        position={[0, -3, 0]}
        material-transparent
        material-opacity={0.3}
      />
    </group>
  );
};

// 3D Customer Journey visualization
const CustomerJourney3D: React.FC<{ journey: CustomerJourney }> = ({ journey }) => {
  const groupRef = useRef<THREE.Group>(null);
  
  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += 0.005;
    }
  });
  
  return (
    <group ref={groupRef}>
      {journey.touchpoints.map((touchpoint, index) => (
        <group key={index} position={touchpoint.position}>
          {/* Touchpoint sphere */}
          <Sphere args={[0.2]}>
            <meshBasicMaterial
              color={touchpoint.event === 'purchase' ? '#39FF14' : '#00FFFF'}
              transparent
              opacity={0.8}
            />
          </Sphere>
          
          {/* Connection lines */}
          {index > 0 && (
            <Line
              points={[
                journey.touchpoints[index - 1].position,
                touchpoint.position
              ]}
              color="#FF1493"
              lineWidth={2}
              transparent
              opacity={0.6}
            />
          )}
          
          {/* Event label */}
          <Text
            position={[0, 0.5, 0]}
            fontSize={0.3}
            color="#FFFFFF"
            anchorX="center"
            anchorY="middle"
          >
            {touchpoint.event}
          </Text>
        </group>
      ))}
    </group>
  );
};

// 3D Scatter plot for customer segmentation
const Scatter3D: React.FC<{ data: Array<{ x: number; y: number; z: number; category: string }> }> = ({ data }) => {
  const groupRef = useRef<THREE.Group>(null);
  
  const colorMap = {
    'Champions': '#39FF14',
    'Loyal': '#00FFFF',
    'At Risk': '#FF1493',
    'Lost': '#FF0040',
    'New': '#FFFF00'
  };
  
  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += 0.002;
    }
  });
  
  return (
    <group ref={groupRef}>
      {data.map((point, index) => (
        <Sphere
          key={index}
          position={[point.x, point.y, point.z]}
          args={[0.15]}
        >
          <meshBasicMaterial
            color={colorMap[point.category as keyof typeof colorMap] || '#FFFFFF'}
            transparent
            opacity={0.7}
          />
        </Sphere>
      ))}
      
      {/* Axis lines */}
      <Line points={[[-5, 0, 0], [5, 0, 0]]} color="#666666" lineWidth={1} />
      <Line points={[[0, -5, 0], [0, 5, 0]]} color="#666666" lineWidth={1} />
      <Line points={[[0, 0, -5], [0, 0, 5]]} color="#666666" lineWidth={1} />
      
      {/* Axis labels */}
      <Text position={[5.5, 0, 0]} fontSize={0.4} color="#FFFFFF">RFM Score</Text>
      <Text position={[0, 5.5, 0]} fontSize={0.4} color="#FFFFFF">LTV</Text>
      <Text position={[0, 0, 5.5]} fontSize={0.4} color="#FFFFFF">Engagement</Text>
    </group>
  );
};

// Cyberpunk lighting setup
const CyberpunkLighting: React.FC = () => {
  return (
    <>
      <ambientLight intensity={0.2} color="#0A0A0A" />
      <pointLight position={[10, 10, 10]} intensity={0.8} color="#00FFFF" />
      <pointLight position={[-10, -10, -10]} intensity={0.6} color="#FF1493" />
      <pointLight position={[0, 10, -10]} intensity={0.4} color="#39FF14" />
      <spotLight
        position={[0, 20, 0]}
        angle={0.3}
        penumbra={1}
        intensity={0.5}
        color="#00FFFF"
        castShadow
      />
    </>
  );
};

// Main HolographicRenderer component
export const HolographicRenderer: React.FC<HolographicRendererProps> = ({
  data,
  type,
  theme,
  interactive = true,
  showParticles = true,
  autoRotate = false
}) => {
  const [viewMode, setViewMode] = useState<'3d' | '2d'>('3d');
  const [showGrid, setShowGrid] = useState(true);
  const [particleCount, setParticleCount] = useState(100);
  
  const renderVisualization = () => {
    switch (type) {
      case 'timeSeries':
        return <HolographicTimeSeries data={data as TimeSeriesData[]} />;
      case 'customerJourney':
        return <CustomerJourney3D journey={data as CustomerJourney} />;
      case 'scatter3d':
        // Convert time series to scatter data for demo
        const scatterData = (data as TimeSeriesData[]).map((point, index) => ({
          x: Math.random() * 10 - 5,
          y: Math.random() * 10 - 5,
          z: Math.random() * 10 - 5,
          category: ['Champions', 'Loyal', 'At Risk', 'New'][index % 4]
        }));
        return <Scatter3D data={scatterData} />;
      default:
        return <HolographicTimeSeries data={data as TimeSeriesData[]} />;
    }
  };
  
  return (
    <CanvasContainer
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.8 }}
    >
      {interactive && (
        <ControlsOverlay>
          <ControlButton onClick={() => setViewMode(viewMode === '3d' ? '2d' : '3d')}>
            {viewMode === '3d' ? '2D' : '3D'}
          </ControlButton>
          <ControlButton onClick={() => setShowGrid(!showGrid)}>
            Grid: {showGrid ? 'ON' : 'OFF'}
          </ControlButton>
          <ControlButton onClick={() => setParticleCount(particleCount === 100 ? 200 : 100)}>
            Particles: {particleCount}
          </ControlButton>
        </ControlsOverlay>
      )}
      
      <Canvas
        camera={{ position: [0, 0, 10], fov: 75 }}
        style={{ background: 'transparent' }}
      >
        <CyberpunkLighting />
        
        {showParticles && (
          <ParticleField count={particleCount} color="#00FFFF" />
        )}
        
        {renderVisualization()}
        
        {interactive && (
          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            autoRotate={autoRotate}
            autoRotateSpeed={0.5}
          />
        )}
      </Canvas>
    </CanvasContainer>
  );
};

// Export additional components for use in other parts of the app
export { ParticleField, CyberpunkLighting };

// Default export
export default HolographicRenderer;