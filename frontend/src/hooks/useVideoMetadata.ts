import { useQuery } from '@tanstack/react-query';
import { fetchVideoMetadata } from '../lib/api';

export function useVideoMetadata(videoId: string | null) {
  return useQuery({
    queryKey: ['video', videoId],
    queryFn: () => fetchVideoMetadata(videoId!),
    enabled: !!videoId,
  });
}
