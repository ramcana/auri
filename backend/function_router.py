import os
import importlib.util
import json

MODULES_DIR = os.path.join(os.path.dirname(__file__), 'modules')

class FunctionRouter:
    def __init__(self):
        self.modules = self.load_modules()

    def load_modules(self):
        modules = {}
        for module_name in os.listdir(MODULES_DIR):
            module_path = os.path.join(MODULES_DIR, module_name)
            if os.path.isdir(module_path):
                config_path = os.path.join(module_path, 'config.json')
                functions_path = os.path.join(module_path, 'functions.py')
                if os.path.isfile(config_path) and os.path.isfile(functions_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    # Dynamically import functions.py
                    spec = importlib.util.spec_from_file_location(f"{module_name}_functions", functions_path)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    modules[module_name] = {
                        'config': config,
                        'functions': mod
                    }
        return modules

    def route(self, user_input, **kwargs):
        """
        Route the user_input to the correct module function based on trigger keywords.
        Returns (module_name, function_name, result) or (None, None, None) if no match.
        """
        for module_name, module in self.modules.items():
            triggers = module['config'].get('trigger_keywords', [])
            if any(trigger in user_input.lower() for trigger in triggers):
                # Find callable functions in the module
                for attr in dir(module['functions']):
                    if not attr.startswith('_') and callable(getattr(module['functions'], attr)):
                        func = getattr(module['functions'], attr)
                        # Call with kwargs if any, else no args
                        try:
                            result = func(**kwargs) if kwargs else func()
                        except Exception as e:
                            result = f"Error in {module_name}.{attr}: {e}"
                        return module_name, attr, result
        return None, None, None

# Example usage:
# router = FunctionRouter()
# module, func, result = router.route("get me the latest news")
# print(module, func, result)
