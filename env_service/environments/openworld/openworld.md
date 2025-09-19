1. openworld连接的mcp服务通过 mcp_tool.json 中配置（在openworld_env.py里第10行指定，通过默认的config或者人工传入）
2. openworld的工具调用模版自己写了一个对应解析的system_prompt,也可以修改这个格式
3. 测试的方式是直接运行openworld_env.py
   --目前用的是debug模式，即存下来的max的单次step过程；也可以换成循环的agent 和环境step
   --对应的logger目前是最详细的，可以在mcp_utils关闭logger 
   -- 目前没有query，在init_state里随便写了两个query； 没有评价（见self.eval函数）
4. 服务器测试，是需要在env_sandbox文件夹下运行openworld.sh；然后再运行envservice下的test_openworld
5. 