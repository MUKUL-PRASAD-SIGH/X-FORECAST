import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';

interface DataPoint {
  x: number;
  y: number;
  z: number;
  value: number;
  label?: string;
  color?: string;
}

interface CyberpunkChart3DProps {
  data: DataPoint[];
  width?: number;
  height?: number;
  title?: string;
  showGrid?: boolean;
  animationSpeed?: number;
  glowIntensity?: number;
}

const CyberpunkChart3D: React.FC<CyberpunkChart3DProps> = ({
  data,
  width = 800,
  height = 600,
  title = "3D Pattern Analysis",
  showGrid = true,
  animationSpeed = 0.01,
  glowIntensity = 0.5
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene>();
  const rendererRef = useRef<THREE.WebGLRenderer>();
  const cameraRef = useRef<THREE.PerspectiveCamera>();
  const frameRef = useRef<number>();
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!mountRef.current || data.length === 0) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.set(10, 10, 10);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    rendererRef.current = renderer;

    // Add cyberpunk lighting
    const ambientLight = new THREE.AmbientLight(0x00ffff, 0.3);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xff00ff, 0.8);
    directionalLight.position.set(10, 10, 5);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    const pointLight1 = new THREE.PointLight(0x00ff00, 0.6, 50);
    pointLight1.position.set(-10, 10, 10);
    scene.add(pointLight1);

    const pointLight2 = new THREE.PointLight(0xff0080, 0.6, 50);
    pointLight2.position.set(10, -10, -10);
    scene.add(pointLight2);

    // Create grid if enabled
    if (showGrid) {
      const gridHelper = new THREE.GridHelper(20, 20, 0x00ffff, 0x004444);
      gridHelper.position.y = -5;
      scene.add(gridHelper);

      // Add axis helpers
      const axesHelper = new THREE.AxesHelper(8);
      scene.add(axesHelper);
    }

    // Create data points
    const geometry = new THREE.SphereGeometry(0.2, 16, 16);
    const dataPoints: THREE.Mesh[] = [];

    data.forEach((point, index) => {
      // Create glowing material
      const material = new THREE.MeshPhongMaterial({
        color: point.color || `hsl(${(index * 137.5) % 360}, 70%, 50%)`,
        emissive: point.color || `hsl(${(index * 137.5) % 360}, 70%, 20%)`,
        emissiveIntensity: glowIntensity,
        shininess: 100,
        transparent: true,
        opacity: 0.8
      });

      const sphere = new THREE.Mesh(geometry, material);
      sphere.position.set(point.x, point.y, point.z);
      sphere.castShadow = true;
      sphere.receiveShadow = true;

      // Add glow effect
      const glowGeometry = new THREE.SphereGeometry(0.3, 16, 16);
      const glowMaterial = new THREE.MeshBasicMaterial({
        color: point.color || `hsl(${(index * 137.5) % 360}, 70%, 50%)`,
        transparent: true,
        opacity: 0.2
      });
      const glow = new THREE.Mesh(glowGeometry, glowMaterial);
      glow.position.copy(sphere.position);
      scene.add(glow);

      scene.add(sphere);
      dataPoints.push(sphere);
    });

    // Create connecting lines for trend visualization
    if (data.length > 1) {
      const lineGeometry = new THREE.BufferGeometry();
      const positions = new Float32Array(data.length * 3);
      
      data.forEach((point, index) => {
        positions[index * 3] = point.x;
        positions[index * 3 + 1] = point.y;
        positions[index * 3 + 2] = point.z;
      });

      lineGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      
      const lineMaterial = new THREE.LineBasicMaterial({
        color: 0x00ffff,
        transparent: true,
        opacity: 0.6,
        linewidth: 2
      });

      const line = new THREE.Line(lineGeometry, lineMaterial);
      scene.add(line);
    }

    // Add title
    if (title) {
      // Create title plane
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d')!;
      canvas.width = 512;
      canvas.height = 128;
      
      context.fillStyle = 'rgba(0, 0, 0, 0.8)';
      context.fillRect(0, 0, canvas.width, canvas.height);
      
      context.fillStyle = '#00ffff';
      context.font = 'bold 32px Arial';
      context.textAlign = 'center';
      context.fillText(title, canvas.width / 2, canvas.height / 2 + 10);
      
      const texture = new THREE.CanvasTexture(canvas);
      const titleMaterial = new THREE.MeshBasicMaterial({
        map: texture,
        transparent: true,
        opacity: 0.9
      });
      
      const titleGeometry = new THREE.PlaneGeometry(8, 2);
      const titleMesh = new THREE.Mesh(titleGeometry, titleMaterial);
      titleMesh.position.set(0, 8, 0);
      scene.add(titleMesh);
    }

    // Controls for camera
    let mouseX = 0;
    let mouseY = 0;
    let targetX = 0;
    let targetY = 0;

    const onMouseMove = (event: MouseEvent) => {
      mouseX = (event.clientX - width / 2) / width;
      mouseY = (event.clientY - height / 2) / height;
    };

    mountRef.current.addEventListener('mousemove', onMouseMove);

    // Animation loop
    const animate = () => {
      frameRef.current = requestAnimationFrame(animate);

      // Smooth camera movement
      targetX = mouseX * 0.001;
      targetY = mouseY * 0.001;

      camera.position.x += (targetX * 20 - camera.position.x) * 0.05;
      camera.position.y += (-targetY * 20 - camera.position.y) * 0.05;
      camera.lookAt(scene.position);

      // Rotate data points
      dataPoints.forEach((point, index) => {
        point.rotation.y += animationSpeed * (1 + index * 0.1);
        point.rotation.x += animationSpeed * 0.5;
      });

      // Pulse glow effect
      const time = Date.now() * 0.001;
      scene.children.forEach((child) => {
        if (child instanceof THREE.Mesh && child.material instanceof THREE.MeshBasicMaterial) {
          if (child.material.opacity < 0.5) { // This is a glow sphere
            child.material.opacity = 0.1 + 0.2 * Math.sin(time * 2 + child.position.x);
          }
        }
      });

      renderer.render(scene, camera);
    };

    mountRef.current.appendChild(renderer.domElement);
    animate();
    setIsLoading(false);

    // Cleanup
    return () => {
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current);
      }
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      if (mountRef.current) {
        mountRef.current.removeEventListener('mousemove', onMouseMove);
      }
      renderer.dispose();
    };
  }, [data, width, height, title, showGrid, animationSpeed, glowIntensity]);

  return (
    <div className="relative">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 z-10">
          <div className="text-cyan-400 text-lg animate-pulse">Loading 3D Visualization...</div>
        </div>
      )}
      <div
        ref={mountRef}
        className="border border-cyan-500 rounded-lg overflow-hidden shadow-lg shadow-cyan-500/20"
        style={{ width, height }}
      />
      <div className="absolute bottom-2 right-2 text-xs text-cyan-400 opacity-70">
        Move mouse to rotate • Cyberpunk 3D Chart
      </div>
    </div>
  );
};

export default CyberpunkChart3D;