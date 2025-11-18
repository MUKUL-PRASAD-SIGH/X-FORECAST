import React, { useState, useEffect, useCallback } from 'react';

interface ProgressUpdate {
  job_id: string;
  stage: string;
  progress_percentage: number;
  message: string;
  timestamp: string;
  model_name?: string;
  details?: any;
}

interface TrainingNotification {
  notification_id: string;
  job_id: string;
  type: 'info' | 'success' | 'warning' | 'error' | 'progress';
  title: string;
  message: string;
  timestamp: string;
  data?: any;
  read: boolean;
}

interface ActiveJob {
  job_id: string;
  company_id: string;
  model_names: string[];
  status: string;
  created_at: string;
  trigger_reason: string;
  current_stage?: string;
  progress_percentage?: number;
  current_message?: string;
}

interface DataQualityResult {
  is_valid: boolean;
  quality_score: number;
  issues: string[];
  warnings: string[];
  recommendations: string[];
  validation_details: any;
}

const TrainingProgressMonitor = () => {
  const [activeJobs, setActiveJobs] = useState<ActiveJob[]>([]);
  const [notifications, setNotifications] = useState<TrainingNotification[]>([]);
  const [selectedJob, setSelectedJob] = useState<string | null>(null);
  const [jobProgress, setJobProgress] = useState<ProgressUpdate[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [loading, setLoading] = useState(true);

  // WebSocket connection for real-time updates
  useEffect(() => {
    const connectWebSocket = () => {
      const ws = new WebSocket(`ws://localhost:8000/api/training-progress/ws`);
      
      ws.onopen = () => {
        setIsConnected(true);
        console.log('Training progress WebSocket connected');
      };
      
      ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        
        if (message.type === 'progress_update') {
          const update = message.data as ProgressUpdate;
          
          // Update job progress if this job is selected
          if (selectedJob === update.job_id) {
            setJobProgress(prev => [...prev, update]);
          }
          
          // Update active jobs
          setActiveJobs(prev => prev.map(job => 
            job.job_id === update.job_id 
              ? {
                  ...job,
                  current_stage: update.stage,
                  progress_percentage: update.progress_percentage,
                  current_message: update.message
                }
              : job
          ));
        } else if (message.type === 'notification') {
          const notification = message.data as TrainingNotification;
          setNotifications(prev => [notification, ...prev.slice(0, 49)]);
        }
      };
      
      ws.onclose = () => {
        setIsConnected(false);
        console.log('Training progress WebSocket disconnected');
        // Attempt to reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
      };
      
      ws.onerror = (error) => {
        console.error('Training progress WebSocket error:', error);
      };
      
      return ws;
    };
    
    const ws = connectWebSocket();
    
    return () => {
      ws.close();
    };
  }, [selectedJob]);

  // Load initial data
  useEffect(() => {
    loadActiveJobs();
    loadNotifications();
  }, []);

  const loadActiveJobs = async () => {
    try {
      const response = await fetch('/api/training-progress/active-jobs');
      const data = await response.json();
      setActiveJobs(data.active_jobs || []);
    } catch (error) {
      console.error('Failed to load active jobs:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadNotifications = async () => {
    try {
      const response = await fetch('/api/training-progress/notifications?limit=20');
      const data = await response.json();
      setNotifications(data || []);
    } catch (error) {
      console.error('Failed to load notifications:', error);
    }
  };

  const loadJobProgress = async (jobId: string) => {
    try {
      const response = await fetch(`/api/training-progress/progress/${jobId}`);
      const data = await response.json();
      setJobProgress(data || []);
    } catch (error) {
      console.error('Failed to load job progress:', error);
    }
  };

  const markNotificationRead = async (notificationId: string) => {
    try {
      await fetch(`/api/training-progress/notifications/${notificationId}/mark-read`, {
        method: 'POST'
      });
      
      setNotifications(prev => 
        prev.map(notif => 
          notif.notification_id === notificationId 
            ? { ...notif, read: true }
            : notif
        )
      );
    } catch (error) {
      console.error('Failed to mark notification as read:', error);
    }
  };

  const handleJobSelect = (jobId: string) => {
    setSelectedJob(jobId);
    loadJobProgress(jobId);
  };

  const getStageIcon = (stage: string) => {
    switch (stage) {
      case 'initializing':
        return '🔄';
      case 'data_validation':
        return '📊';
      case 'pattern_analysis':
        return '🔍';
      case 'model_training':
        return '🤖';
      case 'performance_evaluation':
        return '📈';
      case 'version_comparison':
        return '⚖️';
      case 'activation':
        return '✅';
      case 'completed':
        return '🎉';
      case 'failed':
        return '❌';
      default:
        return '⏳';
    }
  };

  const getNotificationIcon = (type: string) => {
    switch (type) {
      case 'success':
        return '✅';
      case 'error':
        return '❌';
      case 'warning':
        return '⚠️';
      case 'progress':
        return '🔄';
      default:
        return 'ℹ️';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-400 mx-auto mb-4"></div>
          <p className="text-gray-400">Loading training progress...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6 bg-gray-900 min-h-screen text-white">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-cyan-400">Training Progress Monitor</h1>
          <p className="text-gray-400">Real-time monitoring of model training and retraining</p>
        </div>
        <div className="flex items-center space-x-4">
          <div className={`flex items-center space-x-2 px-3 py-1 rounded-full ${
            isConnected ? 'bg-green-900 text-green-400' : 'bg-red-900 text-red-400'
          }`}>
            <div className={`w-2 h-2 rounded-full ${
              isConnected ? 'bg-green-400' : 'bg-red-400'
            }`}></div>
            <span className="text-sm">{isConnected ? 'Connected' : 'Disconnected'}</span>
          </div>
          <button
            onClick={loadActiveJobs}
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 rounded-lg transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Active Jobs */}
        <div className="lg:col-span-2">
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-semibold text-cyan-400 mb-4">Active Training Jobs</h2>
            
            {activeJobs.length === 0 ? (
              <div className="text-center py-8 text-gray-400">
                <p>No active training jobs</p>
              </div>
            ) : (
              <div className="space-y-4">
                {activeJobs.map((job) => (
                  <div
                    key={job.job_id}
                    className={`p-4 rounded-lg border cursor-pointer transition-colors ${
                      selectedJob === job.job_id
                        ? 'border-cyan-400 bg-cyan-900/20'
                        : 'border-gray-600 hover:border-gray-500'
                    }`}
                    onClick={() => handleJobSelect(job.job_id)}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-3">
                        <span className="text-2xl">{getStageIcon(job.current_stage || 'initializing')}</span>
                        <div>
                          <h3 className="font-semibold text-white">Job {job.job_id.slice(-8)}</h3>
                          <p className="text-sm text-gray-400">Company: {job.company_id}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={`px-2 py-1 rounded text-xs ${
                          job.status === 'completed' ? 'bg-green-900 text-green-400' :
                          job.status === 'failed' ? 'bg-red-900 text-red-400' :
                          'bg-blue-900 text-blue-400'
                        }`}>
                          {job.status}
                        </div>
                      </div>
                    </div>
                    
                    <div className="mb-3">
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-400">Progress</span>
                        <span className="text-cyan-400">{job.progress_percentage || 0}%</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div
                          className="bg-cyan-400 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${job.progress_percentage || 0}%` }}
                        ></div>
                      </div>
                    </div>
                    
                    <div className="text-sm text-gray-300 mb-2">
                      {job.current_message || 'Initializing...'}
                    </div>
                    
                    <div className="flex flex-wrap gap-2">
                      {job.model_names.map((model) => (
                        <span
                          key={model}
                          className="px-2 py-1 bg-gray-700 text-gray-300 rounded text-xs"
                        >
                          {model}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Job Progress Details */}
          {selectedJob && (
            <div className="bg-gray-800 rounded-lg p-6 mt-6">
              <h2 className="text-xl font-semibold text-cyan-400 mb-4">
                Progress Details - Job {selectedJob.slice(-8)}
              </h2>
              
              <div className="space-y-3">
                {jobProgress.map((update, index) => (
                  <div key={index} className="flex items-start space-x-3 p-3 bg-gray-700 rounded-lg">
                    <span className="text-xl">{getStageIcon(update.stage)}</span>
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-1">
                        <h4 className="font-medium text-white capitalize">
                          {update.stage.replace('_', ' ')}
                        </h4>
                        <span className="text-sm text-gray-400">
                          {formatTimestamp(update.timestamp)}
                        </span>
                      </div>
                      <p className="text-gray-300 text-sm mb-2">{update.message}</p>
                      {update.model_name && (
                        <span className="px-2 py-1 bg-cyan-900 text-cyan-400 rounded text-xs">
                          {update.model_name}
                        </span>
                      )}
                    </div>
                    <div className="text-right">
                      <span className="text-cyan-400 font-medium">
                        {update.progress_percentage.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Notifications */}
        <div>
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-semibold text-cyan-400 mb-4">Notifications</h2>
            
            {notifications.length === 0 ? (
              <div className="text-center py-8 text-gray-400">
                <p>No notifications</p>
              </div>
            ) : (
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {notifications.map((notification) => (
                  <div
                    key={notification.notification_id}
                    className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                      notification.read
                        ? 'border-gray-600 bg-gray-700/50'
                        : 'border-cyan-400 bg-cyan-900/20'
                    }`}
                    onClick={() => markNotificationRead(notification.notification_id)}
                  >
                    <div className="flex items-start space-x-3">
                      <span className="text-lg">{getNotificationIcon(notification.type)}</span>
                      <div className="flex-1">
                        <h4 className="font-medium text-white text-sm">{notification.title}</h4>
                        <p className="text-gray-300 text-xs mt-1">{notification.message}</p>
                        <p className="text-gray-500 text-xs mt-2">
                          {formatTimestamp(notification.timestamp)}
                        </p>
                      </div>
                      {!notification.read && (
                        <div className="w-2 h-2 bg-cyan-400 rounded-full"></div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainingProgressMonitor;