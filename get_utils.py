import tasks
import predictors


def get_task_class(args):
    if 'iNat' in args.task_name:
        return tasks.iNaturalistMultiTask(args.data_dir, None, args.max_threads, args)
    elif 'CUB' in args.task_name:
        return tasks.CUBMultiTask(args.data_dir, None, args.max_threads, args)
    elif 'Stanford' in args.task_name:
        return tasks.StanfordDogMultiTask(args.data_dir, None, args.max_threads, args)
    elif 'vegfru' in args.task_name:
        return tasks.VegFruMultiTask(args.data_dir, None, args.max_threads, args)
    else:
        raise Exception(f'Unsupported task: {args.task_name}')


def get_predictor(configs):
    if ('iNat' in configs['task_name'] or 'CUB' in configs['task_name'] or 'Stanford' in configs['task_name']
            or 'vegfru' in configs['task_name']):
        return predictors.ThreeClassPredictor(configs)
    else:
        raise Exception(f"Unsupported task: {configs['task_name']}")


def get_exs(args, task):
    train_exs = task.get_even_exs('train', args.n_train)
    if 'CUB' in args.task_name or 'Stanford' in args.task_name:
        val_exs = task.get_even_exs('val', args.n_val)
        test_exs = task.get_even_exs('test', args.n_test)
        return train_exs, val_exs, test_exs
    elif 'iNat' in args.task_name:
        val_exs = None
        test_exs = task.get_even_exs('val', args.n_test)
        return train_exs, val_exs, test_exs
    elif 'vegfru' in args.task_name:
        val_exs = None
        test_exs = task.get_even_exs('test', args.n_test)
        return train_exs, val_exs, test_exs
    else:
        raise Exception(f'Unsupported task: {args.task_name}')
