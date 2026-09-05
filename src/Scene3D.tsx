import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Environment, ContactShadows, RoundedBox } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import * as THREE from 'three';
import { useEffect, useState, type ReactNode } from 'react';

export type DeviceType = 'macbook' | 'iphone' | 'ipad';

interface DeviceCanvasProps {
  device: DeviceType;
  videoSrc: string | null;
  bgColor: string;
  envPreset: string;
  autoRotate: boolean;
  onCreated?: (gl: THREE.WebGLRenderer) => void;
}

// ─── Custom hook: VideoTexture from a URL ────────────────────────────────────

function useVideoTexture(src: string | null): THREE.VideoTexture | null {
  const [texture, setTexture] = useState<THREE.VideoTexture | null>(null);

  useEffect(() => {
    if (!src) { setTexture(null); return; }

    const video = document.createElement('video');
    video.src = src;
    video.loop = true;
    video.muted = true;
    video.playsInline = true;
    video.play().catch(() => {});

    const tex = new THREE.VideoTexture(video);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    setTexture(tex);

    return () => {
      video.pause();
      video.src = '';
      tex.dispose();
      setTexture(null);
    };
  }, [src]);

  // Must be called inside R3F Canvas context
  useFrame(() => { if (texture) texture.needsUpdate = true; });

  return texture;
}

// ─── Shared material props ────────────────────────────────────────────────────

const ALUM = { color: '#B6B6BA', metalness: 0.92, roughness: 0.11, envMapIntensity: 2.2 } as const;
const SCREEN_OFF = '#030C1A';

// ─── MacBook Pro ──────────────────────────────────────────────────────────────

function MacBook({ videoSrc }: { videoSrc: string | null }) {
  const tex = useVideoTexture(videoSrc);

  return (
    <group>
      {/* ── Base ── */}
      <RoundedBox args={[3.6, 0.16, 2.5]} radius={0.07} smoothness={4} position={[0, 0.08, 0]}>
        <meshPhysicalMaterial {...ALUM} />
      </RoundedBox>

      {/* Keyboard deck */}
      <mesh position={[0, 0.162, 0.08]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[3.15, 1.72]} />
        <meshStandardMaterial color="#181818" roughness={0.88} />
      </mesh>

      {/* Trackpad */}
      <RoundedBox args={[0.88, 0.001, 0.54]} radius={0.025} position={[0, 0.163, 0.92]}>
        <meshPhysicalMaterial color="#1E1E22" metalness={0.25} roughness={0.52} />
      </RoundedBox>

      {/* USB-C cutouts (left side) */}
      {[-0.28, -0.48].map((z, i) => (
        <mesh key={i} position={[-1.81, 0.09, z]}>
          <boxGeometry args={[0.006, 0.038, 0.085]} />
          <meshStandardMaterial color="#111114" />
        </mesh>
      ))}

      {/* ── Lid (pivot at back of base) ── */}
      <group position={[0, 0.16, -1.25]} rotation={[-1.88, 0, 0]}>
        {/* Lid exterior */}
        <RoundedBox args={[3.6, 2.32, 0.09]} radius={0.05} smoothness={4} position={[0, 1.16, 0]}>
          <meshPhysicalMaterial {...ALUM} />
        </RoundedBox>

        {/* Inner bezel face */}
        <mesh position={[0, 1.16, 0.047]}>
          <planeGeometry args={[3.52, 2.24]} />
          <meshStandardMaterial color="#090909" roughness={0.6} />
        </mesh>

        {/* ── Screen ── */}
        <mesh position={[0, 1.10, 0.0483]}>
          <planeGeometry args={[3.06, 1.91]} />
          <meshBasicMaterial
            map={tex ?? undefined}
            color={tex ? '#ffffff' : SCREEN_OFF}
            toneMapped={false}
          />
        </mesh>

        {/* Camera dot */}
        <mesh position={[0, 2.20, 0.048]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.019, 0.019, 0.005, 18]} />
          <meshStandardMaterial color="#222225" />
        </mesh>

        {/* Hinge chrome bar */}
        <mesh position={[0, 0.004, 0]}>
          <boxGeometry args={[3.62, 0.013, 0.094]} />
          <meshPhysicalMaterial color="#8A8A92" metalness={0.98} roughness={0.03} />
        </mesh>
      </group>
    </group>
  );
}

// ─── iPhone 15 Pro ────────────────────────────────────────────────────────────

function IPhone15({ videoSrc }: { videoSrc: string | null }) {
  const tex = useVideoTexture(videoSrc);

  return (
    <group>
      {/* Body */}
      <RoundedBox args={[1.12, 2.42, 0.076]} radius={0.12} smoothness={8}>
        <meshPhysicalMaterial color="#1C1C1E" metalness={0.86} roughness={0.08} envMapIntensity={1.8} />
      </RoundedBox>

      {/* Front bezel */}
      <mesh position={[0, 0, 0.0392]}>
        <planeGeometry args={[1.04, 2.34]} />
        <meshStandardMaterial color="#060606" roughness={0.5} />
      </mesh>

      {/* ── Screen ── */}
      <mesh position={[0, -0.016, 0.0400]}>
        <planeGeometry args={[0.96, 2.10]} />
        <meshBasicMaterial
          map={tex ?? undefined}
          color={tex ? '#ffffff' : SCREEN_OFF}
          toneMapped={false}
        />
      </mesh>

      {/* Dynamic Island */}
      <mesh position={[0, 1.025, 0.0408]} rotation={[0, 0, Math.PI / 2]}>
        <capsuleGeometry args={[0.022, 0.22, 4, 16]} />
        <meshStandardMaterial color="#000000" />
      </mesh>

      {/* Buttons */}
      <mesh position={[0.564, 0.28, 0]}>
        <boxGeometry args={[0.009, 0.27, 0.052]} />
        <meshPhysicalMaterial color="#1A1A1C" metalness={0.9} roughness={0.1} />
      </mesh>
      {[0.18, -0.10].map((y, i) => (
        <mesh key={i} position={[-0.564, y, 0]}>
          <boxGeometry args={[0.009, 0.20, 0.042]} />
          <meshPhysicalMaterial color="#1A1A1C" metalness={0.9} roughness={0.1} />
        </mesh>
      ))}
      <mesh position={[-0.564, 0.52, 0]}>
        <boxGeometry args={[0.009, 0.11, 0.038]} />
        <meshPhysicalMaterial color="#1A1A1C" metalness={0.9} roughness={0.1} />
      </mesh>

      {/* Home bar */}
      <mesh position={[0, -1.092, 0.0401]}>
        <planeGeometry args={[0.36, 0.018]} />
        <meshBasicMaterial color="#404044" />
      </mesh>
    </group>
  );
}

// ─── iPad Pro ─────────────────────────────────────────────────────────────────

function IPadPro({ videoSrc }: { videoSrc: string | null }) {
  const tex = useVideoTexture(videoSrc);

  return (
    <group>
      {/* Body */}
      <RoundedBox args={[2.44, 3.3, 0.065]} radius={0.055} smoothness={6}>
        <meshPhysicalMaterial color="#1D1D1F" metalness={0.88} roughness={0.08} envMapIntensity={1.6} />
      </RoundedBox>

      {/* Front bezel */}
      <mesh position={[0, 0, 0.0336]}>
        <planeGeometry args={[2.38, 3.24]} />
        <meshStandardMaterial color="#060606" roughness={0.55} />
      </mesh>

      {/* ── Screen ── */}
      <mesh position={[0, 0, 0.0342]}>
        <planeGeometry args={[2.24, 3.06]} />
        <meshBasicMaterial
          map={tex ?? undefined}
          color={tex ? '#ffffff' : SCREEN_OFF}
          toneMapped={false}
        />
      </mesh>

      {/* Front camera */}
      <mesh position={[0, 1.59, 0.0346]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.025, 0.025, 0.005, 20]} />
        <meshStandardMaterial color="#252528" />
      </mesh>

      {/* Home indicator */}
      <mesh position={[0, -1.5, 0.0343]}>
        <planeGeometry args={[0.42, 0.02]} />
        <meshBasicMaterial color="#333336" />
      </mesh>

      {/* Power button (top) */}
      <mesh position={[0.08, 1.657, 0]}>
        <boxGeometry args={[0.18, 0.009, 0.04]} />
        <meshPhysicalMaterial color="#1D1D1F" metalness={0.88} roughness={0.08} />
      </mesh>

      {/* Volume buttons (right) */}
      {[0.38, 0.10].map((y, i) => (
        <mesh key={i} position={[1.227, y, 0]}>
          <boxGeometry args={[0.009, 0.22, 0.04]} />
          <meshPhysicalMaterial color="#1D1D1F" metalness={0.88} roughness={0.08} />
        </mesh>
      ))}
    </group>
  );
}

// ─── Camera / scene config per device ─────────────────────────────────────────

const DEVICE_CAMERA: Record<DeviceType, { position: [number, number, number]; fov: number }> = {
  macbook: { position: [0.4, 2.2, 6.2], fov: 42 },
  iphone:  { position: [0, 0, 3.8],     fov: 44 },
  ipad:    { position: [0, 0, 5.5],     fov: 44 },
};

// ─── Main Canvas export ───────────────────────────────────────────────────────

export function DeviceCanvas({
  device, videoSrc, bgColor, envPreset, autoRotate, onCreated,
}: DeviceCanvasProps) {
  const cam = DEVICE_CAMERA[device];

  // Per-device model wrappers to keep vertical alignment consistent
  const deviceNode: ReactNode =
    device === 'macbook' ? <MacBook videoSrc={videoSrc} /> :
    device === 'iphone'  ? <group position={[0, 0, 0]}><IPhone15 videoSrc={videoSrc} /></group> :
                           <group position={[0, 0, 0]}><IPadPro  videoSrc={videoSrc} /></group>;

  return (
    <Canvas
      key={device}   // remount on device change → reset camera
      gl={{
        preserveDrawingBuffer: true,
        antialias: true,
        powerPreference: 'high-performance',
        toneMapping: THREE.ACESFilmicToneMapping,
        toneMappingExposure: 1.05,
      }}
      camera={{ fov: cam.fov, position: cam.position, near: 0.1, far: 100 }}
      shadows
      onCreated={({ gl }) => onCreated?.(gl)}
    >
      <color attach="background" args={[bgColor]} />

      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 8, 5]}  intensity={1.5} castShadow shadow-mapSize={[2048, 2048]} />
      <directionalLight position={[-4, 4, -3]} intensity={0.5} color="#8899FF" />
      <pointLight       position={[0, 6, 2]}   intensity={0.3} />

      {/* HDR environment for reflections */}
      <Environment preset={envPreset as 'apartment'} environmentIntensity={0.9} />

      {deviceNode}

      {/* Soft drop shadow */}
      <ContactShadows
        position={[0, device === 'macbook' ? -0.001 : -1.5, 0]}
        opacity={0.45}
        scale={device === 'macbook' ? 12 : 6}
        blur={2.5}
        far={5}
      />

      {/* Camera controls */}
      <OrbitControls
        autoRotate={autoRotate}
        autoRotateSpeed={1.5}
        dampingFactor={0.06}
        enableDamping
        minPolarAngle={0.05}
        maxPolarAngle={Math.PI / 2 - 0.05}
        target={device === 'macbook' ? [0, 0.6, 0] : [0, 0, 0]}
      />

      {/* Subtle screen glow via bloom */}
      <EffectComposer>
        <Bloom luminanceThreshold={0.45} luminanceSmoothing={0.5} intensity={0.3} levels={6} />
      </EffectComposer>
    </Canvas>
  );
}
