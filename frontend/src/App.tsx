import './App.css';
import { useEffect } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useStore } from './stores/useStore';
import { useVideos } from './hooks/useVideos';
import { useVideoMetadata } from './hooks/useVideoMetadata';
import { useTracklets } from './hooks/useTracklets';
import Header from './components/Header/Header';
import EmbeddingsPanel from './components/EmbeddingsPanel/EmbeddingsPanel';
import RightPanel from './components/RightPanel/RightPanel';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function AppContent() {
  const {
    selectedVideoId,
    setVideos,
    setVideoMetadata,
    setTracklets,
  } = useStore();

  const { data: videos } = useVideos();
  const { data: videoMetadata } = useVideoMetadata(selectedVideoId);
  const { data: tracklets } = useTracklets(selectedVideoId, videoMetadata?.total_tracklets);

  useEffect(() => {
    if (videos) setVideos(videos);
  }, [videos, setVideos]);

  useEffect(() => {
    setVideoMetadata(videoMetadata ?? null);
  }, [videoMetadata, setVideoMetadata]);

  useEffect(() => {
    setTracklets(tracklets ?? []);
  }, [tracklets, setTracklets]);

  return (
    <div className="flex flex-col overflow-hidden" style={{ height: '100dvh' }}>
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <div className="relative overflow-hidden border-r border-gray-700" style={{ width: '45%' }}>
          <EmbeddingsPanel />
        </div>
        <div className="relative overflow-hidden" style={{ width: '55%' }}>
          <RightPanel />
        </div>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppContent />
    </QueryClientProvider>
  );
}
