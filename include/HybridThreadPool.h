#include <functional>
#include <future>
#include <queue>
#include <vector>
enum class DeviceType { CPU, GPU };

struct Task {
    DeviceType deviceType;
    std::function<void()> cpuTask;
    std::function<void()> gpuTask;
};

class HybridThreadPool {
public:
    HybridThreadPool(size_t cpuThreads, size_t gpuThreads);
    ~HybridThreadPool();

    template<class F, class... Args>
    auto enqueue(DeviceType preferredDevice, F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;

private:
    std::vector<std::thread> cpuWorkers;
    std::vector<std::thread> gpuWorkers;
    std::queue<Task> tasks;
    
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

    void cpuWorkerThread();
    void gpuWorkerThread();
};

HybridThreadPool::HybridThreadPool(size_t cpuThreads, size_t gpuThreads)
    : stop(false)
{
    for (size_t i = 0; i < cpuThreads; ++i)
        cpuWorkers.emplace_back(&HybridThreadPool::cpuWorkerThread, this);
    
    for (size_t i = 0; i < gpuThreads; ++i)
        gpuWorkers.emplace_back(&HybridThreadPool::gpuWorkerThread, this);
}

void HybridThreadPool::cpuWorkerThread() {
    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            condition.wait(lock, [this] { return stop || !tasks.empty(); });
            if (stop && tasks.empty()) return;
            task = std::move(tasks.front());
            tasks.pop();
        }
        if (task.deviceType == DeviceType::CPU || task.gpuTask == nullptr)
            task.cpuTask();
    }
}

void HybridThreadPool::gpuWorkerThread() {
    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            condition.wait(lock, [this] { return stop || !tasks.empty(); });
            if (stop && tasks.empty()) return;
            task = std::move(tasks.front());
            tasks.pop();
        }
        if (task.deviceType == DeviceType::GPU && task.gpuTask != nullptr)
            task.gpuTask();
        else
            task.cpuTask();
    }
}

template<class F, class... Args>
auto HybridThreadPool::enqueue(DeviceType preferredDevice, F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace(Task{
            preferredDevice,
            [task](){ (*task)(); },
            nullptr  // 这里可以添加 GPU 版本的任务
        });
    }
    condition.notify_one();
    return res;
}