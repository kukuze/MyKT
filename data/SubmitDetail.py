class SubmitDetail:
    def __init__(self, cid, cfid, qindex, submitTime, difficulty, tags, name, content, status):
        self.cid = cid
        self.cfid = cfid
        self.qindex = qindex
        self.difficulty = difficulty
        self.name = name
        self.tags = tags
        self.status = status
        self.submitTime = submitTime
        self.content = content
    def __getitem__(self, key):
        return getattr(self, key)

    @classmethod
    def from_dict(cls, data):
        return cls(
            cid=data['cid'],
            cfid=data['cfid'],
            qindex=data['qindex'],
            submitTime=data['submitTime'],
            difficulty=data['difficulty'],
            tags=data['tags'],
            name=data['name'],
            content=data['content'],
            status=data['status']
        )

    def to_dict(self):
        return {
            'cid': self.cid,
            'cfid': self.cfid,
            'qindex': self.qindex,
            'submitTime': self.submitTime,
            'difficulty': self.difficulty,
            'tags': self.tags,
            'name': self.name,
            'content': self.content,
            'status': self.status
        }
