# Function to render an agent in a specific gym enviroment in external GUI -> not so laggy like matplotlib implementation
import numpy as np

def render_GUI(render_env, render_agent, filepath_actor, filepath_critic):

    render_agent.load_models(filepath_actor, filepath_critic)
    render_agent._init_networks()
    render_obs,_ = render_env.reset()

    # Start rendering in pyglet GUI (internal gym method which uses pyglet inthe background)
    render_env.render()
    episode = 0

    #Run the GUI until Keyboard Interrupt hits (only cell-interrupt in Jupyter-Notebook is supported. It's not possible to close the GUI directly!)
    try:
        while True:
            render_action = render_agent.act(np.array([render_obs]))
            _, _, termination, truncation, _ = render_env.step(render_action)

            if termination or truncation:
                print(f'Episode {episode} finished')
                episode += 1
                render_obs,_ = render_env.reset()

    except KeyboardInterrupt as e:
        print('Closed Rendering sucessful')
        render_env.close()